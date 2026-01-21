# src/ingestion/service.py
import os
import hashlib
import config
import concurrent.futures
from tqdm import tqdm
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_batch
from indexing.faiss_index import reset_all_indexes, load_all_indexes, save_all_indexes
from indexing.metadata_index import init_db, clear_metadata, get_all_metadata
from utils.label_detector import analyze_dataset_structure
from utils.logger import setup_logger

logger = setup_logger("IngestionService")

def calculate_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(config.FILE_READ_BUFFER_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception: return None

def _worker_load_file(args):
    file_path, valid_labels = args
    try:
        file_hash = calculate_file_hash(file_path)
        docs = dispatch_loader(file_path, valid_labels=valid_labels)
        if not docs: return []
        
        for i, doc in enumerate(docs):
            doc['source'] = str(file_path)
            if len(docs) > 1:
                unique_sig = f"{file_hash}_{i}"
                doc['file_hash'] = hashlib.md5(unique_sig.encode()).hexdigest()
            else:
                doc['file_hash'] = file_hash
        return docs
    except Exception: return []

class IngestionService:
    @staticmethod
    def prepare_database(mode='r'):
        if mode == 'r':
            reset_all_indexes()
            clear_metadata() 
            logger.info("Base de données réinitialisée.")
        else:
            init_db() 
            load_all_indexes()
            logger.info("Base de données chargée pour complétion.")
            
    @staticmethod
    def get_files_to_ingest(mode='r'):
        if not os.path.exists(config.DATASET_DIR):
            raise FileNotFoundError(f"Dossier source introuvable : {config.DATASET_DIR}")
        all_files = scan_folder(config.DATASET_DIR)
        if mode == 'c':
            processed_sources = {m['source'] for m in get_all_metadata()}
            return [f for f in all_files if f not in processed_sources]
        return all_files

    @staticmethod
    def run_workflow(mode='r'):
        cpu_count = os.cpu_count() or 1
        MAX_WORKERS = min(cpu_count - 2, 28)

        logger.info(f"--- Workflow Industriel (Streaming Mode) ---")
        
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        total_indexed = 0
        stream_buffer = [] 

        try:
            tasks = [(f, valid_labels) for f in files_to_process]
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results_gen = executor.map(_worker_load_file, tasks, chunksize=config.INGESTION_CHUNKSIZE)
                
                pbar = tqdm(total=total_files, desc="Streaming Ingestion")
                
                for res in results_gen:
                    if res:
                        stream_buffer.extend(res)
                    
                    if len(stream_buffer) >= config.BATCH_SIZE:
                        total_indexed += process_batch(stream_buffer, valid_labels)
                        stream_buffer = [] 
                    
                    pbar.update(1)

                if stream_buffer:
                    total_indexed += process_batch(stream_buffer, valid_labels)
                pbar.close()

        except KeyboardInterrupt:
            logger.warning("\nInterruption détectée.")
        finally:
            if total_indexed > 0:
                save_all_indexes()
                logger.info(f"Workflow terminé : {total_indexed} documents indexés.")
            
        return total_indexed, total_files