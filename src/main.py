# src/main.py
import sys
from src.utils.logger import setup_logger
import argparse
import uvicorn
from pathlib import Path
from utils.environment import check_environment
from utils.logger import setup_logger

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


logger = setup_logger("Orchestrator")

def get_ingestion_service():
    from src.ingestion.service import IngestionService
    return IngestionService()

def main():
    # --- 1. PRE-FLIGHT CHECK ---
    check_environment()

    # --- 2. GESTION DU CLI ---
    parser = argparse.ArgumentParser(description="SmartSearchEngine CLI")
    parser.add_argument(
        "mode", 
        choices=["ingest", "serve"], 
        help="Lancer l'ingestion (ingest) ou le serveur API (serve)"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # --- 3. EXÉCUTION DU MODE ---
    if args.mode == "ingest":
        logger.info("Mode : Ingestion des données")
        from ingestion.main import main as run_ingestion
        run_ingestion()

    elif args.mode == "serve":
        logger.info(" Mode : Lancement du serveur Search API")
        
        uvicorn.run(
            "search.main:app", 
            host="localhost", 
            port=8000, 
            reload=True
        )

if __name__ == "__main__":
    main()