# src/utils/environment.py
import importlib
from src import config  
from src.utils.logger import setup_logger 

logger = setup_logger("EnvChecker")

def check_environment():
    logger.info("Lancement des vérifications de l'environnement (Architecture LanceDB)...")
    
    # 1. Vérification des dossiers de base
    if not config.DATASET_DIR.exists():
        logger.warning(f"Dossier dataset introuvable : {config.DATASET_DIR}")
        config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Dossier dataset créé.")

    # 2. Vérification des moteurs IA & Données
    required_modules = {
        "lancedb": "Moteur de stockage VectorStore",
        "pyarrow": "Gestionnaire de format de données Arrow",
        "paddleocr": "Moteur OCR (PaddleOCR)",
        "fitz": "Lecteur PDF (PyMuPDF)"
    }

    for module, description in required_modules.items():
        try:
            importlib.import_module(module)
            logger.info(f"{description} détecté.")
        except ImportError:
            if module in ["lancedb", "pyarrow"]:
                logger.error(f"CRITIQUE : {module} n'est pas installé. L'application ne peut pas fonctionner.")
            else:
                logger.warning(f" {description} ({module}) n'est pas installé.")


    config.COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    config.LANCEDB_URI.mkdir(parents=True, exist_ok=True)
    
    logger.info(" Structure de l'application validée (LanceDB Store).")
    logger.info(f"Matériel utilisé : {config.DEVICE}")
    return True