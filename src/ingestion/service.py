# src/ingestion/service.py
import os
import hashlib
import concurrent.futures
import psutil
import torch 
from tqdm import tqdm
from src.utils.spinner import TqdmHeartbeat
from src import config
from src.ingestion.folder_scanner import scan_folder
from src.ingestion.dispatcher import dispatch_loader
from src.ingestion.core import process_batch
from src.indexing.vector_store import (
    init_tables, reset_store, check_file_status, 
    update_file_source, create_vector_index
)
from src.utils.label_detector import analyze_dataset_structure
from src.utils.preprocessing import calculate_fast_hash
from src.utils.logger import setup_logger
logger = setup_logger("IngestionService")

def _init_ocr_worker():
    """Initialise PaddleOCR sur CPU pour chaque worker (Sécurité Deadlock/OOM)."""
    from paddleocr import PaddleOCR
    global local_ocr
    local_ocr = PaddleOCR(lang=config.OCR_LANG, use_angle_cls=True, show_log=False, use_gpu=False)

def _worker_load_file(args):
    """Worker léger : OCR CPU uniquement."""
    file_path, file_hash = args
    try:
        docs = dispatch_loader(file_path, valid_labels=None)
        if not docs: return []
        
        for i, doc in enumerate(docs):
            doc['source'] = str(file_path)
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
    def get_files_to_ingest(mode='r'):
        """Version Elite : Fast-Check en mémoire pour une vitesse foudroyante."""
        all_paths = scan_folder(config.DATASET_DIR)
        
        if mode == 'r':
            return [(f, calculate_fast_hash(f)) for f in all_paths]

        from src.indexing.vector_store import get_all_indexed_hashes
        indexed_hashes = get_all_indexed_hashes() 
        
        to_process = []
        skipped = 0
        
        for f in tqdm(all_paths, desc="Fast-Check Incremental"):
            f_hash = calculate_fast_hash(f)
            if not f_hash: continue
            
            # Vérification instantanée en RAM (Set lookup)
            if f_hash in indexed_hashes:
                skipped += 1
            else:
                to_process.append((f, f_hash))
                
        logger.info(f"Fast-Check : {skipped} déjà indexés, {len(to_process)} nouveaux à traiter.")
        return to_process

    @staticmethod
    def run_workflow(mode='r'):
        """Orchestre le Workflow de Ingestion à Indexation."""
        logger.info(f"--- Workflow LanceDB Démarré (Batch: {config.BATCH_SIZE}) ---")
        
        # 1. Préparation du Store 
        if mode == 'r':
            reset_store()
        else:
            init_tables()

        # 2. Analyse de structure 
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_info = IngestionService.get_files_to_ingest(mode)
        
        if not files_info:
            logger.info("Aucun nouveau document à traiter.")
            return 0, 0

        total_indexed = 0
        stream_buffer = []
        
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=config.MAX_WORKERS,
                initializer=_init_ocr_worker
            ) as executor:
                
                results_gen = executor.map(_worker_load_file, files_info, chunksize=config.INGESTION_CHUNKSIZE)
                pbar = tqdm(total=len(files_info), desc=" Streaming Ingestion")
                heartbeat = TqdmHeartbeat(pbar, "Streaming Ingestion")
                heartbeat.start()

                try:
                    for res in results_gen:
                        if res: stream_buffer.extend(res)
                        
                        while len(stream_buffer) >= config.BATCH_SIZE:
                            current_batch = stream_buffer[:config.BATCH_SIZE]
                            stream_buffer = stream_buffer[config.BATCH_SIZE:]
                            total_indexed += process_batch(current_batch, valid_labels)
                            
                            pbar.set_postfix({
                                "RAM": f"{psutil.virtual_memory().percent}%", 
                                "Vectors": total_indexed
                            })
                        
                        pbar.update(1)

                    # Traitement du reliquat
                    if stream_buffer:
                        total_indexed += process_batch(stream_buffer, valid_labels)

                finally:
                    heartbeat.stop()
                    pbar.close()

        except KeyboardInterrupt:
            logger.warning("\nInterruption manuelle.")
        except Exception as e:
            logger.error(f"Erreur critique durant le streaming : {e}")
        finally:
            # 3. ÉTAPE FINALE ÉLITE : Création de l'index disque
            if total_indexed > 0:
                logger.info("Optimisation de l'index vectoriel sur disque...")
                create_vector_index()
                logger.info(f"Workflow terminé : {total_indexed} documents synchronisés.")
            
        return total_indexed, len(files_info)