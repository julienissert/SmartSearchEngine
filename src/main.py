# src/main.py
import sys
import os
import multiprocessing
import argparse

# --- CORRECTIF CRITIQUE GPU ---
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# --- IMPORTS ÉLITE (Tous préfixés par src.) ---
from src.utils.logger import setup_logger
from src.services.watcher import start_watching
from src.services.environment import check_environment

logger = setup_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="Moteur de Recherche Multimodal Élite")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Commande Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Lancer l'ingestion des données")
    ingest_parser.add_argument("-m", "--mode", choices=['r', 'c'], default='c', 
                                help="r: reset (réinitialiser), c: compléter")

    # Commande Watch
    subparsers.add_parser("watch", help="Lancer la surveillance en temps réel")

    # Commande Serve
    subparsers.add_parser("serve", help="Démarrer l'API de recherche")

    args = parser.parse_args()

    # Vérification de l'environnement
    if not check_environment():
        logger.error("Environnement invalide. Arrêt.")
        sys.exit(1)

    if args.command == "ingest":
        from src.ingestion.main import run_ingestion_logic
        run_ingestion_logic(mode=args.mode)
    
    elif args.command == "watch":
        logger.info(" Lancement du mode surveillance...")
        start_watching()
        
    elif args.command == "serve":
        import uvicorn
        logger.info("Démarrage de l'API sur http://localhost:8000")
        uvicorn.run("src.search.main:app", host="0.0.0.0", port=8000, reload=False)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()