# src/ingestion/service.py
import os
import config
import concurrent.futures
from tqdm import tqdm
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_batch
from indexing.faiss_index import (
    reset_all_indexes, load_all_indexes, save_all_indexes
)
from indexing.metadata_index import (
    save_metadata_to_disk, clear_metadata, load_metadata_from_disk, get_all_metadata
)
from utils.label_detector import analyze_dataset_structure
from utils.logger import setup_logger

logger = setup_logger("IngestionService")

def _worker_load_file(args):
    file_path, valid_labels = args
    try:
        # 1. Parsing du fichier
        docs = dispatch_loader(file_path, valid_labels=valid_labels)
        if not docs: 
            return []
        
        for doc in docs:
            if not doc.get('source'):
                doc['source'] = str(file_path)
                
        return docs
    except Exception:
        return []

class IngestionService:
    @staticmethod
    def prepare_database(mode='r'):
        if mode == 'r':
            reset_all_indexes()
            clear_metadata() 
            logger.info("Base de données réinitialisée (Reset mode).")
        else:
            load_metadata_from_disk()
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
        # --- 1. RESSOURCES ---
        cpu_count = os.cpu_count() or 1
        MAX_WORKERS = min(cpu_count - 2, 28)

        logger.info(f"--- Workflow Standard (Batch Mode) ---")
        
        # --- 2. PRÉPARATION ---
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        new_docs_count = 0
        all_docs_buffer = []

        # --- 3. CHARGEMENT  ---
        try:
            tasks = [(f, valid_labels) for f in files_to_process]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(_worker_load_file, tasks, chunksize=config.INGESTION_CHUNKSIZE), 
                    total=total_files, 
                    desc="Parsing des fichiers"
                ))

                for res in results:
                    if res:
                        all_docs_buffer.extend(res)

            # --- 4. TRAITEMENT PAR BATCH ---
            if all_docs_buffer:
                logger.info(f"Début de l'indexation de {len(all_docs_buffer)} documents...")
                

                process_batch(all_docs_buffer, valid_labels)
                
                new_docs_count = len(all_docs_buffer)

        except KeyboardInterrupt:
            logger.warning("\nInterruption. Sauvegarde de sécurité...")
        
        finally:
            if new_docs_count > 0:
                save_metadata_to_disk()
                save_all_indexes()
                logger.info(f"Workflow terminé : {new_docs_count} documents sécurisés.")
            
        return new_docs_count, total_files