# src/main.py
import sys
import os
import multiprocessing

# --- CORRECTIF CRITIQUE GPU ---
# Force les workers à démarrer "à neuf" (Spawn) au lieu de cloner le parent (Fork).
# Cela empêche les workers d'hériter du contexte CUDA et de saturer la VRAM.
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

from utils.environment import check_environment
from utils.logger import setup_logger

logger = setup_logger("Main")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/main.py [ingest|watch|serve] [options]")
        sys.exit(1)

    command = sys.argv[1]

    # Vérification environnementale avant tout
    if not check_environment():
        logger.error("Environnement invalide. Arrêt.")
        sys.exit(1)

    if command == "ingest":
        from ingestion.main import run_ingestion_logic
        run_ingestion_logic()
    
    elif command == "watch":
        from utils.watcher import start_watcher
        start_watcher()
        
    elif command == "serve":
        import uvicorn
        # Lancement de l'API
        uvicorn.run("search.main:app", host="0.0.0.0", port=8000, reload=False)
        
    else:
        logger.error(f"Commande inconnue: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()