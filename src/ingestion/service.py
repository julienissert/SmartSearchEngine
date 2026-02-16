# src/ingestion/service.py
import os
import hashlib
import concurrent.futures
import json
import gc
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from src import config
from src.config import monitor
from src.utils.logger import setup_logger
from src.utils.preprocessing import calculate_fast_hash, calculate_folder_signature
from src.interface.spinner import TqdmHeartbeat
from src.ingestion.folder_scanner import scan_folder
from src.ingestion.dispatcher import dispatch_loader, VISUAL_EXTENSIONS
from src.intelligence.label_detector import analyze_dataset_structure, clear_memory
from src.ingestion.core import process_batch
from src.indexing.vector_store import (
    init_tables, reset_store, create_vector_index,
    get_folder_contract, save_folder_contract, get_all_indexed_hashes
)
from paddleocr import PaddleOCR

logger = setup_logger("IngestionService")

# État global du worker
_WORKER_CONTEXT = {}
local_ocr = None

def _init_worker(context):
    """Initialise PaddleOCR et injecte le contexte une seule fois par worker."""
    global local_ocr, _WORKER_CONTEXT
    # OCR sur CPU pour la stabilité Windows
    local_ocr = PaddleOCR(lang=config.OCR_LANG, use_angle_cls=True, show_log=False, use_gpu=False)
    _WORKER_CONTEXT = context

def _worker_load_file(args):
    """Tâche légère : Charge le contenu brut (Texte/OCR)."""
    file_path, file_hash = args
    try:
        # dispatch_loader utilise context pour l'arbitrage visuel/label
        docs = dispatch_loader(file_path, valid_labels=_WORKER_CONTEXT)
        if not docs: return []
        
        for i, doc in enumerate(docs):
            doc['source'] = str(file_path)
            doc['file_hash'] = hashlib.md5(f"{file_hash}_{i}".encode()).hexdigest() if len(docs) > 1 else file_hash
        return docs
    except Exception as e:
        logger.error(f" Erreur worker sur {os.path.basename(file_path)} : {e}")
        return []

class IngestionService:
    @staticmethod
    def get_grouped_files(mode='r'):
        """Scan hiérarchique avec saut de dossier (O(1)) et analyse delta."""
        dataset_path = config.DATASET_DIR
        archives = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
        
        indexed_hashes = get_all_indexed_hashes() if mode != 'r' else set()
        grouped_to_process = defaultdict(list)
        skipped_archives, skipped_files = 0, 0

        pbar = tqdm(archives, desc=" Fast-Check Datasets")
        heartbeat = TqdmHeartbeat(pbar, "Scanning")
        heartbeat.start()

        try:
            for arch_path in pbar:
                current_sig = calculate_folder_signature(arch_path) # Signature instantanée
                contract = get_folder_contract(arch_path)
                
                if mode != 'r' and contract and contract.get('signature') == current_sig:
                    skipped_archives += 1
                    continue
                
                files_in_arch = scan_folder(arch_path)
                for f in files_in_arch:
                    f_hash = calculate_fast_hash(f)
                    if not f_hash or (mode != 'r' and f_hash in indexed_hashes):
                        skipped_files += 1
                        continue
                    grouped_to_process[arch_path].append((f, f_hash, current_sig))
        finally:
            heartbeat.stop()
            pbar.close()
                
        logger.info(f"Optimisation : {skipped_archives} dossiers ignorés | {skipped_files} fichiers évités.")
        return grouped_to_process

    @staticmethod
    def run_workflow(mode='r'):
        if mode == 'r': reset_store()
        else: init_tables()
            
        grouped_files = IngestionService.get_grouped_files(mode)
        if not grouped_files: return 0, 0
        
        total_indexed = 0

        for archive_path, files_info in grouped_files.items():
            archive_name = os.path.basename(archive_path)
            _, _, folder_sig = files_info[0] 
            logger.info(f"\n>>> Traitement Dataset : {archive_name}")

            # --- AJOUT : Initialisation des variables de suivi pour ce dossier ---
            detected_domain = "unknown"
            best_confidence = 0.0

            # 1. Analyse IA et plans
            context = analyze_dataset_structure(archive_path)
            plans = context.get('file_plans', {})
            image_map = context.get('image_map', {})
            resolved_images_to_skip = set()

            # 2. Skip-List optimisée (Liaisons Texte-Image)
            if image_map and plans:
                for f_path, plan in plans.items():
                    p_key = plan.get('path_key')
                    if not p_key: continue
                    try:
                        ext = f_path.lower()
                        if ext.endswith('.json'):
                            with open(f_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                records = data if isinstance(data, list) else [data]
                                col_data = [str(r.get(p_key, "")).strip().lower() for r in records]
                        else:
                            sep = '\t' if ext.endswith('.tsv') else (',' if ext.endswith('.csv') else None)
                            df = pd.read_csv(f_path, sep=sep, engine='python', on_bad_lines='skip', usecols=[p_key])
                            col_data = df[p_key].dropna().astype(str).str.strip().str.lower().tolist()

                        for img_name in col_data:
                            full_img = image_map.get(img_name)
                            if full_img: resolved_images_to_skip.add(os.path.abspath(full_img).lower())
                    except Exception as e:
                        logger.warning(f" Erreur Skip-List sur {os.path.basename(f_path)} : {e}")

            if resolved_images_to_skip:
                logger.info(f" [LIAISON DÉTECTÉE] {len(resolved_images_to_skip)} images réservées pour la fusion.")

            # 3. Exécution avec gestion de flux sécurisée
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=monitor.get_max_workers(),
                initializer=_init_worker,
                initargs=(context,)
            ) as executor:

                tasks = [(f, h) for f, h, _ in files_info if os.path.abspath(f).lower() not in resolved_images_to_skip]
                results_gen = executor.map(_worker_load_file, tasks, chunksize=1)
                
                pbar = tqdm(total=len(tasks), desc=f" {archive_name[:15]}")
                heartbeat = TqdmHeartbeat(pbar, archive_name[:15])
                heartbeat.start()
                stream_buffer = []

                for docs in results_gen:
                    pbar.update(1)
                    if not docs: continue
                    for doc in docs:
                        # ... (votre code existant pour le smart linking) ...
                        stream_buffer.append(doc)
                        
                        if len(stream_buffer) >= config.BATCH_SIZE:
                            monitor.throttle() 
                            # --- MODIFICATION : On capture le domaine et le score ---
                            count, domain, score = process_batch(stream_buffer, context) 
                            
                            if domain != "unknown":
                                detected_domain = domain
                                best_confidence = score

                            total_indexed += count
                            pbar.update(len(stream_buffer))
                            stream_buffer = []

                # --- MODIFICATION : On capture aussi les infos pour le flush final ---
                while stream_buffer:
                    chunk = stream_buffer[:config.BATCH_SIZE]
                    count, domain, score = process_batch(chunk, context)
                    
                    if domain != "unknown":
                        detected_domain = domain
                        best_confidence = score

                    total_indexed += count
                    pbar.update(len(chunk))
                    stream_buffer = stream_buffer[config.BATCH_SIZE:]

                # --- MODIFICATION : On enregistre le VRAI domaine détecté ---
                save_folder_contract(archive_path, detected_domain, folder_sig, best_confidence)
                
                heartbeat.stop()
                pbar.close()
                clear_memory()

        if total_indexed > 0: create_vector_index()
        return total_indexed, sum(len(v) for v in grouped_files.values())