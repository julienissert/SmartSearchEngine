# src/ingestion/service.py
import os
import config
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_document
from indexing.faiss_index import reset_all_indexes
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
        """Initialise ou charge la base de données selon le mode."""
        if mode == 'r':
            reset_all_indexes()
            clear_metadata()
            logger.info("Base de données réinitialisée (Reset mode).")
            return "Base réinitialisée"
        else:
            load_metadata_from_disk()
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
            files = [f for f in all_files if f not in processed_sources]
            logger.info(f"Mode complétion : {len(files)} nouveaux fichiers détectés sur {len(all_files)}.")
            return files
        
        logger.info(f"Mode réinitialisation : {len(all_files)} fichiers à traiter.")
        return all_files

    @staticmethod
    def run_workflow(mode='r'):
        logger.info("--- Démarrage d'un nouveau workflow d'ingestion ---")
        
        # 1. Préparation
        IngestionService.prepare_database(mode)
        
        # 2. Analyse structurelle
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        
        # 3. Récupération des fichiers
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        # 4. Traitement
        new_docs_count = 0
        for f in files_to_process:
            try:
                for doc in dispatch_loader(f):
                    process_document(doc, valid_labels)
                    new_docs_count += 1
            except Exception as e:
                logger.error(f"Erreur critique sur le fichier {f} : {str(e)}")
                continue 

        # 5. Sauvegarde
        if new_docs_count > 0:
            save_metadata_to_disk()
            logger.info(f"Sauvegarde réussie : {new_docs_count} nouveaux documents indexés.")
            
        return new_docs_count, len(files_to_process)