# src/services/environment.py
import importlib
from src import config  
from src.utils.logger import setup_logger 
# On importe l'instance singleton du LLM Manager
from src.intelligence.llm_manager import llm

logger = setup_logger("EnvChecker")

def check_environment():
    """
    Vérifie l'intégrité du système : Dossiers, Dépendances, IA (Ollama) et Stockage.
    """
    logger.info("Lancement des vérifications de l'environnement (Architecture LanceDB)...")
    
    all_systems_go = True

    # 1. Vérification des dossiers de base
    # On s'assure que l'arborescence critique existe
    critical_dirs = [config.DATASET_DIR, config.COMPUTED_DIR, config.LANCEDB_URI]
    for directory in critical_dirs:
        if not directory.exists():
            logger.warning(f"Création du dossier manquant : {directory}")
            directory.mkdir(parents=True, exist_ok=True)

    # 2. Vérification des moteurs IA & Données (Modules Python)
    required_modules = {
        "lancedb": "Moteur de stockage VectorStore",
        "pyarrow": "Gestionnaire de format de données Arrow",
        "paddleocr": "Moteur OCR (PaddleOCR)",
        "fitz": "Lecteur PDF (PyMuPDF)"
    }

    for module, description in required_modules.items():
        try:
            importlib.import_module(module)
            logger.info(f"✅ {description} détecté.")
        except ImportError:
            if module in ["lancedb", "pyarrow"]:
                logger.error(f"❌ CRITIQUE : {module} n'est pas installé. L'application ne peut pas fonctionner.")
                all_systems_go = False
            else:
                logger.warning(f"⚠️ {description} ({module}) n'est pas installé. Certaines fonctions seront limitées.")

    # 3. VÉRIFICATION OLLAMA (Via le singleton LLMManager)
    # Note : Le LLMManager a déjà tenté de se connecter lors de son import
    if not llm.is_healthy():
        logger.warning("⚠️ Ollama n'est pas détecté ou ne répond pas. L'intelligence sera désactivée.")
        # On ne bloque pas forcément l'app, mais on prévient
    else:
        # CORRECTION ICI : On utilise llm.model au lieu de config.LLM_MODEL pour éviter l'AttributeError
        logger.info(f"✅ Ollama est opérationnel (Modèle actif : {llm.model})")
        
    logger.info("Structure de l'application validée (LanceDB Store).")
    logger.info(f"Matériel utilisé pour l'inférence : {config.DEVICE}")
    
    return all_systems_go