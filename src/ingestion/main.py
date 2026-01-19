# src/ingestion/main.py
import time
from src import config
from src.ingestion.service import IngestionService
from src.utils.logger import setup_logger

logger = setup_logger("IngestionService")

def run_ingestion_logic(mode=None):
    start_time = time.time()

    # Mode interactif si lancé sans argument -m
    if mode is None:
        if config.METADATA_DIR.exists() and any(config.METADATA_DIR.iterdir()):
            choice = input("\nBase existante. (R)éinitialiser ou (C)ompléter ? [R/C] : ").lower()
            mode = 'c' if choice == 'c' else 'r'
        else:
            mode = 'r'

    try:
        service = IngestionService()
        new_docs, total_files = service.run_workflow(mode)
        duration = time.time() - start_time
        logger.info(f"Terminé : {new_docs} docs indexés en {duration:.2f}s.")
    except Exception as e:
        logger.error(f"Erreur fatale ingestion : {e}")