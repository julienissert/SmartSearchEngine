# src/ingestion/service.py
import os
import config
from tqdm import tqdm  
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_batch 
from indexing.faiss_index import (
    reset_all_indexes, 
    load_all_indexes, 
    save_all_indexes  
)
from indexing.metadata_index import (
    save_metadata_to_disk, 
    clear_metadata, 
    load_metadata_from_disk, 
    get_all_metadata
)
from utils.label_detector import analyze_dataset_structure
from utils.logger import setup_logger

logger = setup_logger("IngestionService")

class IngestionService:
    @staticmethod
    def prepare_database(mode='r'):
        if mode == 'r':
            reset_all_indexes()
            clear_metadata()
            logger.info("Base de données réinitialisée (Reset mode).")
            return "Base réinitialisée"
        else:
            load_metadata_from_disk()
            load_all_indexes()
            logger.info("Base de données chargée pour complétion.")
            return "Base chargée"

    @staticmethod
    def get_files_to_ingest(mode='r'):
        if not os.path.exists(config.DATASET_DIR):
            logger.error(f"Dossier source introuvable : {config.DATASET_DIR}")
            raise FileNotFoundError(f"Dossier source introuvable : {config.DATASET_DIR}")
        all_files = scan_folder(config.DATASET_DIR)
        if mode == 'c':
            processed_sources = {m['source'] for m in get_all_metadata()}
            return [f for f in all_files if f not in processed_sources]
        return all_files

    @staticmethod
    def run_workflow(mode='r'):
        """Workflow industriel avec traitement par lots et RAM-First indexing."""
        logger.info("--- Démarrage du workflow d'ingestion (Mode Batch) ---")
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        new_docs_count = 0
        
        for f in tqdm(files_to_process, desc="Total Fichiers", unit="file"):
            try:
                docs = dispatch_loader(f, valid_labels=valid_labels)
                
                desc_intern = f" > {os.path.basename(f)[:20]}"
                for i in range(0, len(docs), config.BATCH_SIZE):
                    batch = docs[i : i + config.BATCH_SIZE]
                    process_batch(batch, valid_labels)
                    new_docs_count += len(batch)
                    
            except Exception as e:
                logger.error(f"Erreur critique sur le fichier {f} : {str(e)}")
                continue 

        if new_docs_count > 0:
            save_metadata_to_disk()
            save_all_indexes()
            logger.info(f"Sauvegarde réussie : {new_docs_count} nouveaux documents indexés.")
            
        return new_docs_count, len(files_to_process)