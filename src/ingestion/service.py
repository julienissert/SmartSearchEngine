# src/ingestion/service.py
import os
import hashlib
import concurrent.futures
import psutil
import torch 
from tqdm import tqdm

from src import config
from src.ingestion.folder_scanner import scan_folder
from src.ingestion.dispatcher import dispatch_loader
from src.ingestion.core import process_batch
from src.indexing.faiss_index import reset_all_indexes, load_all_indexes, save_all_indexes
from src.indexing.metadata_index import (
    init_db, clear_metadata, check_file_status, update_file_source
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
        """RESTAURÉ : Logique de Fast-Check pour l'ingestion incrémentale."""
        all_paths = scan_folder(config.DATASET_DIR)
        if mode == 'r':
            return [(f, calculate_fast_hash(f)) for f in all_paths]

        to_process = []
        skipped, moved = 0, 0
        for f in tqdm(all_paths, desc="Fast-Check Incremental"):
            f_hash = calculate_fast_hash(f)
            if not f_hash: continue
            
            status = check_file_status(f_hash, f)
            if status == 'exists': skipped += 1
            elif status == 'moved':
                update_file_source(f_hash, f)
                moved += 1
            else: to_process.append((f, f_hash))
                
        logger.info(f"Fast-Check : {skipped} inchangés, {moved} déplacés, {len(to_process)} à traiter.")
        return to_process

    @staticmethod
    def run_workflow(mode='r'):
        """Orchestre le Workflow : Muscles (Performance) + Cerveau (Sémantique)."""
        logger.info(f"--- Workflow Démarré (Workers: {config.MAX_WORKERS} | Batch: {config.BATCH_SIZE}) ---")
        
        # 1. Préparation DB
        if mode == 'r':
            reset_all_indexes()
            clear_metadata()
        else:
            init_db()
            load_all_indexes()

        # 2. Analyse de structure 
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_info = IngestionService.get_files_to_ingest(mode)
        
        if not files_info:
            logger.info("Aucun nouveau document à traiter.")
            return 0, 0

        total_indexed = 0
        stream_buffer = []

        # 3. Streaming Ingestion avec protection matérielle
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=config.MAX_WORKERS,
                initializer=_init_ocr_worker
            ) as executor:
                
                results_gen = executor.map(_worker_load_file, files_info, chunksize=config.INGESTION_CHUNKSIZE)
                pbar = tqdm(total=len(files_info), desc=" Streaming Ingestion")
                
                for res in results_gen:
                    if res: stream_buffer.extend(res)
                    
                    # ---  BATCHING STRICT (Protection VRAM RTX) ---
                    while len(stream_buffer) >= config.BATCH_SIZE:
                        current_batch = stream_buffer[:config.BATCH_SIZE]
                        stream_buffer = stream_buffer[config.BATCH_SIZE:]
                        
                        total_indexed += process_batch(current_batch, valid_labels)
                        
                        if config.DEVICE == "cuda":
                            torch.cuda.empty_cache() 
                            
                        pbar.set_postfix({"RAM": f"{psutil.virtual_memory().percent}%", "Vectors": total_indexed})
                    
                    pbar.update(1)

                if stream_buffer:
                    total_indexed += process_batch(stream_buffer, valid_labels)

                pbar.close()

        except KeyboardInterrupt:
            logger.warning("\nInterruption manuelle.")
        finally:
            if total_indexed > 0:
                save_all_indexes()
                logger.info(f"Terminé : {total_indexed} nouveaux documents indexés.")
            
        return total_indexed, len(files_info)