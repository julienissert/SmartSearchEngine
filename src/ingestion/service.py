# src/ingestion/service.py
import os
import config
import concurrent.futures
import psutil
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

# Seuil de sauvegarde pour éviter la saturation RAM (ex: tous les 5000 docs)
SAVE_INTERVAL = 5000 

def _worker_load_file(args):
    file_path, valid_labels = args
    try:
        return dispatch_loader(file_path, valid_labels=valid_labels)
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
        # --- 1. CONFIGURATION RESSOURCES ---
        cpu_count = os.cpu_count() or 1
        CHUNK_SIZE = 50 
        MAX_WORKERS = max(1, cpu_count - 4)

        logger.info(f"--- Démarrage du workflow HAUTE DISPONIBILITÉ ---")

        # --- 2. PRÉPARATION ---
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        new_docs_count = 0        # Total pour le rapport final
        unsaved_docs_count = 0    # Compteur pour le prochain checkpoint
        current_batch = []

        # --- 3. EXÉCUTION STREAMING AVEC CHECKPOINTS ---
        try:
            tasks = [(f, valid_labels) for f in files_to_process]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Itération streaming
                progress_bar = tqdm(
                    executor.map(_worker_load_file, tasks, chunksize=CHUNK_SIZE), 
                    total=total_files, 
                    desc="Ingestion Massive", 
                    unit="file"
                )

                for docs in progress_bar:
                    if not docs: continue
                    current_batch.extend(docs)

                    # A. Si le batch CLIP est plein, on vectorise
                    if len(current_batch) >= config.BATCH_SIZE:
                        batch_size = len(current_batch)
                        process_batch(current_batch, valid_labels)
                        
                        new_docs_count += batch_size
                        unsaved_docs_count += batch_size
                        current_batch = [] 

                        # B. STRATÉGIE ANTI-SATURATION : Sauvegarde périodique sur disque
                        if unsaved_docs_count >= SAVE_INTERVAL:
                            logger.info(f" Checkpoint : Sauvegarde de {unsaved_docs_count} docs sur disque...")
                            save_metadata_to_disk()
                            save_all_indexes()
                            unsaved_docs_count = 0 # On remet à zéro après le flush
                            
                            # Log de la RAM pour monitoring
                            ram_usage = psutil.virtual_memory().percent
                            progress_bar.set_postfix({"RAM": f"{ram_usage}%", "Total": new_docs_count})

            # Traitement du dernier batch restant
            if current_batch:
                process_batch(current_batch, valid_labels)
                new_docs_count += len(current_batch)

        except KeyboardInterrupt:
            logger.warning("\nInterruption. Sauvegarde de sécurité...")
        
        finally:
            # --- 4. FINALISATION ---
            # On sauvegarde le reliquat qui n'a pas atteint le dernier SAVE_INTERVAL
            if new_docs_count > 0:
                save_metadata_to_disk()
                save_all_indexes()     
                logger.info(f"Workflow terminé. Total indexé : {new_docs_count} documents.")
            
        return new_docs_count, total_files