# src/main.py
import sys
import argparse
import uvicorn
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
# On récupère le chemin absolu du dossier 'src'
SRC_DIR = Path(__file__).resolve().parent
# On récupère le chemin de la racine du projet (au-dessus de 'src')
ROOT_DIR = SRC_DIR.parent

# On ajoute les DEUX dossiers au path pour supporter tous les styles d'imports
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Maintenant, 'import config' ET 'from src import config' fonctionneront
from utils.environment import check_environment
from utils.logger import setup_logger

logger = setup_logger("Orchestrator")

def main():
    check_environment() #

    parser = argparse.ArgumentParser(description="SmartSearchEngine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commandes")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Lancer l'ingestion")
    ingest_parser.add_argument("-m", "--mode", choices=['r', 'c'], help="r: reset, c: compléter")

    # Serve
    subparsers.add_parser("serve", help="Lancer l'API")

    # Watch
    subparsers.add_parser("watch", help="Lancer le Watcher")

    args = parser.parse_args()

    if args.command == "ingest":
        from ingestion.main import run_ingestion_logic
        run_ingestion_logic(mode=args.mode)
    elif args.command == "serve":
        logger.info("Lancement Search API")
        uvicorn.run("search.main:app", host="0.0.0.0", port=8000, reload=True)
    elif args.command == "watch":
        from utils.watcher import start_watching
        start_watching()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()