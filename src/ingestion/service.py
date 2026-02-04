# src/ingestion/service.py
import os
import hashlib
import concurrent.futures
import psutil
import gc
import torch 
from tqdm import tqdm
from src.interface.spinner import TqdmHeartbeat
from src import config
from src.ingestion.folder_scanner import scan_folder
from src.ingestion.dispatcher import dispatch_loader
from src.intelligence.label_detector import analyze_dataset_structure, clear_memory
from src.ingestion.core import process_batch
from src.indexing.vector_store import (
    init_tables, reset_store, 
    create_vector_index,
    get_folder_contract, save_folder_contract
)
from src.utils.preprocessing import calculate_fast_hash,calculate_folder_signature
from src.utils.logger import setup_logger
from src.indexing.vector_store import get_all_indexed_hashes
from collections import defaultdict
from src.config import monitor
from paddleocr import PaddleOCR

logger = setup_logger("IngestionService")

def _init_ocr_worker():
    """Initialise PaddleOCR sur CPU pour chaque worker (Sécurité Deadlock/OOM)."""
    global local_ocr
    local_ocr = PaddleOCR(lang=config.OCR_LANG, use_angle_cls=True, show_log=False, use_gpu=False)

def _worker_load_file(args):
    """Worker ultra-léger : OCR CPU uniquement, pas de transfert d'image."""
    file_path, file_hash, context = args
    try:
        # dispatch_loader extrait le texte (OCR)
        docs = dispatch_loader(file_path, valid_labels=context)
        if not docs: return []
        
        for i, doc in enumerate(docs):
            doc['source'] = str(file_path)
            doc['image'] = None 
            
            if 'extra' not in doc:
                doc['extra'] = {}
            if len(docs) > 1:
                doc['file_hash'] = hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest()
            else:
                doc['file_hash'] = file_hash
        return docs
    except Exception:
        return []

class IngestionService:
    @staticmethod
    def get_grouped_files(mode='r'):
        """
        NIVEAU 1 : Saut de dossier par Signature (Hyper-Vitesse).
        NIVEAU 2 : Fast-Check par Hash de fichier (Analyse Delta).
        """        
        # Identification des dossiers racines (archives)
        dataset_path = config.DATASET_DIR
        archives = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
        
        indexed_hashes = get_all_indexed_hashes() if mode != 'r' else set()
        grouped_to_process = defaultdict(list)
        
        skipped_archives = 0
        skipped_files = 0

        # Progress bar pour le scan initial
        pbar = tqdm(archives, desc=" Fast-Check & Hierarchical Grouping")
        heartbeat = TqdmHeartbeat(pbar, "Scanning Datasets")
        heartbeat.start()

        try:
            for arch_path in pbar:
                arch_name = os.path.basename(arch_path)
                
                # --- NIVEAU 1 : Signature du Dossier (0.001s) ---
                current_sig = calculate_folder_signature(arch_path)
                contract = get_folder_contract(arch_path)
                
                # Si mode 'c' et que la signature correspond : ON SAUTE TOUT LE DOSSIER
                if mode != 'r' and contract and contract.get('signature') == current_sig:
                    skipped_archives += 1
                    continue
                
                # --- NIVEAU 2 : Analyse Delta (Fichier par fichier) ---
                files_in_arch = scan_folder(arch_path)
                
                for f in files_in_arch:
                    f_hash = calculate_fast_hash(f)
                    if not f_hash: continue
                    
                    if mode != 'r' and f_hash in indexed_hashes:
                        skipped_files += 1
                        continue
                    
                    grouped_to_process[arch_path].append((f, f_hash, current_sig))
        finally:
            heartbeat.stop()
            pbar.close()
                
        logger.info(
            f"Optimisation : {skipped_archives} dossiers ignorés (signatures identiques). "
            f" Analyse Delta : {skipped_files} fichiers existants évités dans les dossiers modifiés."
        )
        return grouped_to_process

    @staticmethod
    def run_workflow(mode='r'):
        if mode == 'r': reset_store()
        else: init_tables()
            
        grouped_files = IngestionService.get_grouped_files(mode)
        if not grouped_files:
            logger.info("Base de données à jour. Rien à ingérer.")
            return 0, 0
        
        total_indexed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.MAX_WORKERS,
            initializer=_init_ocr_worker
        ) as executor:

            for archive_path, files_info in grouped_files.items():
                archive_name = os.path.basename(archive_path)
                _, _, folder_sig = files_info[0] 
                
                # --- VARIABLES POUR LE CONTRAT ---
                detected_domain = "unknown"
                detected_score = 0.0

                logger.info(f"\n>>> Traitement Dataset : {archive_name} ({len(files_info)} nouveaux fichiers)")

                valid_labels = analyze_dataset_structure(archive_path)
                stream_buffer = []

                worker_tasks = [(f, h, valid_labels) for f, h, s in files_info]
                results_gen = executor.map(_worker_load_file, worker_tasks, chunksize=config.INGESTION_CHUNKSIZE)

                pbar = tqdm(total=len(files_info), desc=f" {archive_name[:15]}")
                heartbeat = TqdmHeartbeat(pbar, f"Ingestion {archive_name[:10]}")
                heartbeat.start()

                try:
                    for res in results_gen:
                        if res: stream_buffer.extend(res)
                        monitor.throttle()
                        
                        while len(stream_buffer) >= config.BATCH_SIZE:
                            current_batch = stream_buffer[:config.BATCH_SIZE]
                            stream_buffer = stream_buffer[config.BATCH_SIZE:]
                            
                            # --- RÉCEPTION DU TRIPLET ---
                            count, b_domain, b_score = process_batch(current_batch, valid_labels)
                            total_indexed += count
                            
                            # On mémorise la détection la plus fraîche de l'IA
                            if b_domain != "unknown":
                                detected_domain, detected_score = b_domain, b_score
                            
                            pbar.set_postfix({"RAM": f"{psutil.virtual_memory().percent}%", "Totals": total_indexed})
                        pbar.update(1)

                    # --- CORRECTIF RELIQUAT = ---
                    if stream_buffer:
                        count, b_domain, b_score = process_batch(stream_buffer, valid_labels)
                        total_indexed += count
                        if b_domain != "unknown":
                            detected_domain, detected_score = b_domain, b_score

                    # --- L'EMPLACEMENT DU SAVE (Retiré du core) ---
                    final_domain = detected_domain
                    if final_domain == "unknown" and isinstance(valid_labels, dict):
                        final_domain = valid_labels.get('domain', 'unknown')
                        
                    is_verified = 1 if detected_score >= 0.90 else 0
                    save_folder_contract(
                        folder_path=archive_path, 
                        domain=str(final_domain), 
                        signature=folder_sig, # Empreinte du dossier (Niveau 1)
                        confidence=float(detected_score) if detected_score > 0 else 1.0,
                        verified=is_verified
                    )
                    logger.info(f" Contrat scellé : {final_domain} ({detected_score:.2f})")

                except Exception as e:
                    logger.error(f" Erreur durant le traitement de {archive_name} : {e}")
                finally:
                    heartbeat.stop()
                    pbar.close()
                    clear_memory() 

        if total_indexed > 0:
            create_vector_index()
            
        return total_indexed, sum(len(v) for v in grouped_files.values())