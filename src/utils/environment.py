# src/utils/environment.py
import config
from utils.logger import setup_logger

logger = setup_logger("EnvChecker")

def check_environment():
    logger.info("Lancement des vérifications de l'environnement...")
    
    # 1. Vérification des dossiers de base
    if not config.DATASET_DIR.exists():
        logger.warning(f"Dossier dataset introuvable : {config.DATASET_DIR}")
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Dossier dataset créé.")

    # 2. Vérification de Tesseract OCR
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        logger.info("Tesseract OCR détecté.")
    except Exception:
        logger.error("Tesseract OCR n'est pas installé ou n'est pas dans le PATH.")
        logger.error("L'ingestion d'images (OCR) ne fonctionnera pas correctement.")

    # 3. Validation de la structure des dossiers générés
    config.COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True) 
    
    logger.info("Structure de l'application validée.")
    return True