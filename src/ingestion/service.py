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

class IngestionService:
    @staticmethod
    def prepare_database(mode='r'):
        """Initialise ou charge la base de données selon le mode (Reset ou Complete)."""
        if mode == 'r':
            reset_all_indexes()
            clear_metadata()
            return "Base réinitialisée"
        else:
            load_metadata_from_disk()
            return "Base chargée pour complétion"

    @staticmethod
    def get_files_to_ingest(mode='r'):
        """Scanne et filtre les fichiers à traiter."""
        if not os.path.exists(config.DATASET_DIR):
            raise FileNotFoundError(f"Dossier source introuvable : {config.DATASET_DIR}")

        all_files = scan_folder(config.DATASET_DIR)
        
        if mode == 'c':
            processed_sources = {m['source'] for m in get_all_metadata()}
            return [f for f in all_files if f not in processed_sources]
        
        return all_files

    @staticmethod
    def run_workflow(mode='r'):
        """Exécute la logique complète d'ingestion et retourne le nombre de documents ajoutés."""
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
            except Exception:
                continue 

        # 5. Sauvegarde
        if new_docs_count > 0:
            save_metadata_to_disk()
            
        return new_docs_count, len(files_to_process)