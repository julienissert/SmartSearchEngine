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

def _worker_load_file(args):
    """Fonction exécutée par les workers pour le parsing et l'OCR."""
    file_path, valid_labels = args
    try:
        # L'OCR s'exécute ici dans les processus enfants
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
        # --- 1. DÉTECTION DES RESSOURCES ---
        cpu_count = os.cpu_count() or 1
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Optimisation : 100k fichiers demandent un chunksize plus élevé pour réduire l'IPC overhead
        CHUNK_SIZE = 50 
        
        env_workers = os.getenv("MAX_WORKERS")
        if env_workers:
            MAX_WORKERS = int(env_workers)
            selection_mode = "Manuel"
        else:
            # Sur 30 cœurs, on laisse de la place pour le processus principal (Vectorisation)
            MAX_WORKERS = max(1, cpu_count - 4)
            selection_mode = "Auto-Optimisé"

        logger.info(f"--- Démarrage du workflow STREAMING ---")
        logger.info(f"Workers OCR : {MAX_WORKERS} | Chunksize : {CHUNK_SIZE}")

        # --- 2. PRÉPARATION ---
        IngestionService.prepare_database(mode)
        # L'analyse est maintenant batchée (voir label_detector.py)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        new_docs_count = 0
        current_batch = []

        # --- 3. EXÉCUTION EN STREAMING (CPU OCR + GPU/CPU Vectorisation) ---
        try:
            tasks = [(f, valid_labels) for f in files_to_process]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # IMPORTANT : On itère DIRECTEMENT sur l'itérateur sans le transformer en liste
                # Cela permet de traiter le fichier 1 pendant que le worker traite le fichier 10
                progress_bar = tqdm(
                    executor.map(_worker_load_file, tasks, chunksize=CHUNK_SIZE), 
                    total=total_files, 
                    desc="Workflow Multimodal (OCR -> CLIP -> FAISS)", 
                    unit="file"
                )

                for docs in progress_bar:
                    if not docs: continue
                    current_batch.extend(docs)

                    # Si le batch est prêt, on vectorise immédiatement
                    if len(current_batch) >= config.BATCH_SIZE:
                        process_batch(current_batch, valid_labels)
                        new_docs_count += len(current_batch)
                        current_batch = [] 
                        progress_bar.set_postfix({"indexed": new_docs_count})

            # Reliquat final
            if current_batch:
                process_batch(current_batch, valid_labels)
                new_docs_count += len(current_batch)

        except KeyboardInterrupt:
            logger.warning("\nInterruption détectée. Sauvegarde des données déjà traitées...")
        except Exception as e:
            logger.error(f"Erreur critique : {e}", exc_info=True)
        
        finally:
            if new_docs_count > 0:
                save_metadata_to_disk()
                save_all_indexes()     
                logger.info(f"Sauvegarde réussie : {new_docs_count} documents.")
            
        return new_docs_count, total_files