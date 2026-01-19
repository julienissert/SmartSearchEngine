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

# Initialisation du logger avec protection contre les doublons
logger = setup_logger("IngestionService")

def _worker_load_file(args):
    """Fonction exécutée par les workers pour le parsing et l'OCR."""
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
        # 1. Préparation et Analyse
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        new_docs_count = 0
        current_batch = []

        # --- 2. CALCUL ÉLASTIQUE ET ADAPTATIF DES WORKERS ---
        cpu_count = os.cpu_count() or 1
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # Vérification d'une limite manuelle (Priorité utilisateur)
        env_workers = os.getenv("MAX_WORKERS")
        
        if env_workers:
            MAX_WORKERS = int(env_workers)
            selection_mode = "Manuel (Variable d'environnement)"
        else:
            # Calcul automatique de base
            ram_limit = max(1, int(available_ram_gb // 3))
            cpu_limit = max(1, cpu_count - 2)
            
            if config.DEVICE == "cuda":
                # Sécurité GPU : On bride à 2 workers pour éviter le freeze CUDA/Docker
                MAX_WORKERS = min(2, cpu_limit)
                selection_mode = "Auto-Bridé (Sécurité GPU)"
            else:
                # Mode CPU : On utilise les ressources au maximum
                MAX_WORKERS = min(cpu_limit, ram_limit)
                selection_mode = "Automatique (Optimisé CPU)"

        logger.info(f"--- Démarrage du workflow PARALLÈLE ---")
        logger.info(f"Matériel utilisé : {config.DEVICE.upper()}")
        logger.info(f"Ressources : {cpu_count} CPUs, {available_ram_gb:.1f} Go RAM disponible.")
        logger.info(f"Workers actifs : {MAX_WORKERS} ({selection_mode}).")      

        # 3. Exécution Parallèle (OCR & Parsing)
        tasks = [(f, valid_labels) for f in files_to_process]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(
                executor.map(_worker_load_file, tasks, chunksize=1), 
                total=total_files, 
                desc="Ingestion (OCR & Parsing)", 
                unit="file"
            ))

            # 4. Traitement séquentiel (Vectorisation CLIP & Indexation)
            logger.info("Début de la vectorisation et de l'indexation...")
            
            for docs in tqdm(results, desc="Indexation", unit="doc"):
                if not docs: continue
                current_batch.extend(docs)

                if len(current_batch) >= config.BATCH_SIZE:
                    process_batch(current_batch, valid_labels)
                    new_docs_count += len(current_batch)
                    current_batch = [] 

            # Traitement du dernier lot
            if current_batch:
                process_batch(current_batch, valid_labels)
                new_docs_count += len(current_batch)

        # 5. Sauvegarde finale
        if new_docs_count > 0:
            save_metadata_to_disk()
            save_all_indexes()
            logger.info(f"Sauvegarde réussie : {new_docs_count} nouveaux documents indexés.")
            
        return new_docs_count, total_files