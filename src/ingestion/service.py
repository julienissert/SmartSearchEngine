# src/ingestion/service.py
import os
import hashlib
import concurrent.futures
from tqdm import tqdm

import config
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_batch
from indexing.faiss_index import reset_all_indexes, load_all_indexes, save_all_indexes
from indexing.metadata_index import (
    init_db, clear_metadata, check_file_status, update_file_source
)
from utils.label_detector import analyze_dataset_structure
from utils.preprocessing import calculate_fast_hash
from utils.logger import setup_logger

logger = setup_logger("IngestionService")

def _worker_load_file(args):
    """Charge un fichier et prépare ses documents avec le hash pré-calculé."""
    file_path, file_hash, valid_labels = args
    try:
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
    except Exception as e:
        logger.error(f"Erreur worker sur {file_path}: {e}")
        return []
class IngestionService:
    @staticmethod
    def prepare_database(mode='r'):
        if mode == 'r':
            reset_all_indexes()
            clear_metadata() 
            logger.info("Base de données et Index FAISS réinitialisés (Mode Reset).")
        else:
            init_db() 
            load_all_indexes()
            logger.info("Base de données chargée (Mode Complétion).")
            
    @staticmethod
    def get_files_to_ingest(mode='r'):
        if not os.path.exists(config.DATASET_DIR):
            raise FileNotFoundError(f"Dossier source introuvable : {config.DATASET_DIR}")

        all_paths = scan_folder(config.DATASET_DIR)
        
        if mode == 'r':
            return [(f, calculate_fast_hash(f)) for f in all_paths]

        # LOGIQUE FAST-CHECK (Mode Complétion)
        to_process = []
        skipped, moved = 0, 0
        
        for f in tqdm(all_paths, desc="Fast-Check Incremental"):
            f_hash = calculate_fast_hash(f)
            if not f_hash: continue
            
            status = check_file_status(f_hash, f)
            
            if status == 'exists':
                skipped += 1
            elif status == 'moved':
                update_file_source(f_hash, f)
                moved += 1
            else:
                to_process.append((f, f_hash))
                
        logger.info(f"Fast-Check : {skipped} inchangés, {moved} déplacés, {len(to_process)} à traiter.")
        return to_process

    @staticmethod
    def run_workflow(mode='r'):
        """Orchestre le pipeline avec l'intelligence adaptative du config."""       
        logger.info(f"--- Démarrage Workflow (Workers: {config.MAX_WORKERS} | Batch: {config.BATCH_SIZE} | Device: {config.DEVICE}) ---")
        
        IngestionService.prepare_database(mode)
        valid_labels = analyze_dataset_structure(config.DATASET_DIR)
        files_info = IngestionService.get_files_to_ingest(mode)
        
        total_files = len(files_info)
        if total_files == 0:
            logger.info("Aucun nouveau fichier à indexer.")
            return 0, 0

        total_indexed = 0
        stream_buffer = [] 

        try:
            tasks = [(f, h, valid_labels) for f, h in files_info]
            
            # Utilisation de config.MAX_WORKERS pour respecter la RAM
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                
                # Utilisation de config.INGESTION_CHUNKSIZE pour l'équilibre
                results_gen = executor.map(_worker_load_file, tasks, chunksize=config.INGESTION_CHUNKSIZE)
                
                pbar = tqdm(total=total_files, desc="Streaming Ingestion")
                
                for res in results_gen:
                    if res:
                        stream_buffer.extend(res)
                    
                    # Utilisation de config.BATCH_SIZE (variable selon GPU/CPU)
                    if len(stream_buffer) >= config.BATCH_SIZE:
                        total_indexed += process_batch(stream_buffer, valid_labels)
                        stream_buffer = [] # Vidage immédiat pour la RAM
                    
                    pbar.update(1)

                if stream_buffer:
                    total_indexed += process_batch(stream_buffer, valid_labels)
                pbar.close()

        except KeyboardInterrupt:
            logger.warning("\nInterruption manuelle détectée.")
        except Exception as e:
            logger.error(f"Erreur critique lors du workflow : {e}")
        finally:
            if total_indexed > 0:
                save_all_indexes()
                logger.info(f"Succès : {total_indexed} nouveaux documents ajoutés.")
            
        return total_indexed, total_files