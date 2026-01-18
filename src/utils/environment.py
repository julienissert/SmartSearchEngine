# src/utils/environment.py
import config
import importlib
from utils.logger import setup_logger

logger = setup_logger("EnvChecker")

def check_environment():
    logger.info("Lancement des vérifications de l'environnement...")
    
    # 1. Vérification des dossiers de base
    if not config.DATASET_DIR.exists():
        logger.warning(f"Dossier dataset introuvable : {config.DATASET_DIR}")
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Dossier dataset créé.")

    # 2. Vérification des moteurs IA réellement utilisés
    # On vérifie PaddleOCR pour le traitement des images
    try:
        importlib.import_module("paddleocr")
        logger.info("Moteur OCR (PaddleOCR) détecté.")
    except ImportError:
        logger.error("PaddleOCR n'est pas installé. L'ingestion d'images échouera.")

    # On vérifie PyMuPDF (fitz) pour l'extraction de texte des PDF
    try:
        importlib.import_module("fitz") 
        logger.info("Lecteur PDF (PyMuPDF) détecté.")
    except ImportError:
        logger.warning("PyMuPDF n'est pas installé. L'extraction de texte des PDF sera limitée.")

    # 3. Validation de la structure des dossiers de données
    config.COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True) 
    
    logger.info("Structure de l'application validée.")
    logger.info(f"Matériel utilisé pour l'IA : {config.DEVICE}")
    return True