# src/ingestion/service.py
import os
import config
import concurrent.futures
import psutil  # Ajouté pour la gestion intelligente de la RAM
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

# Fonction helper exécutée par les workers (indépendante de la classe)
def _worker_load_file(args):
    file_path, valid_labels = args
    try:
        # Cette étape inclut l'OCR Paddle qui prend du temps CPU et beaucoup de RAM
        return dispatch_loader(file_path, valid_labels=valid_labels)
    except Exception as e:
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
        # 1. Analyse initiale
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_to_process = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_to_process)
        new_docs_count = 0
        current_batch = []
        
        # --- 2. CALCUL ÉLASTIQUE ET SÉCURISÉ DES WORKERS ---
        cpu_count = os.cpu_count() or 1
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        # On estime qu'un worker OCR (Paddle) consomme ~2 Go de RAM
        ram_limit = max(1, int(available_ram_gb // 2))
        # On laisse toujours 2 cœurs libres pour le système et l'orchestrateur
        cpu_limit = max(1, cpu_count - 2)
        
        # On prend le plus petit des deux pour garantir la stabilité
        MAX_WORKERS = min(cpu_limit, ram_limit)
        
        logger.info(f"--- Démarrage du workflow PARALLÈLE ---")
        logger.info(f"Matériel : {cpu_count} CPUs, {available_ram_gb:.1f} Go RAM disponible.")
        logger.info(f"Workers actifs : {MAX_WORKERS} (Optimisé pour éviter les crashs RAM).")

        # 3. Préparation des tâches
        tasks = [(f, valid_labels) for f in files_to_process]

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # On lance le traitement (OCR + Parsing)
            results = list(tqdm(
                executor.map(_worker_load_file, tasks, chunksize=10), 
                total=total_files, 
                desc="Ingestion (OCR & Parsing)", 
                unit="file"
            ))

            # 4. Traitement séquentiel des résultats (Vectorisation CLIP + Indexation)
            logger.info("Début de la vectorisation et de l'indexation...")
            
            for docs in tqdm(results, desc="Indexation", unit="doc"):
                if not docs: continue
                
                current_batch.extend(docs)

                # Vectorisation par lots (BATCH_SIZE) pour optimiser le CPU/GPU
                if len(current_batch) >= config.BATCH_SIZE:
                    process_batch(current_batch, valid_labels)
                    new_docs_count += len(current_batch)
                    current_batch = [] 

            # Reliquat final
            if current_batch:
                process_batch(current_batch, valid_labels)
                new_docs_count += len(current_batch)

        # 5. Sauvegarde physique
        if new_docs_count > 0:
            save_metadata_to_disk()
            save_all_indexes()
            logger.info(f"Sauvegarde réussie : {new_docs_count} nouveaux documents indexés.")
            
        return new_docs_count, total_files