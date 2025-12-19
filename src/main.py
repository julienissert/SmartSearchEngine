# src/main.py
import os
import sys
import argparse
import uvicorn

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

def main():
    parser = argparse.ArgumentParser(description="MAYbe Here - SmartSearchEngine CLI")
    parser.add_argument(
        "mode", 
        choices=["ingest", "serve"], 
        help="Lancer l'ingestion des données (ingest) ou le serveur de recherche (serve)"
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.mode == "ingest":
        print("Mode : Ingestion des données en cours...")
        from ingestion.main import main as run_ingestion
        run_ingestion()

    elif args.mode == "serve":
        print("Mode : Lancement du serveur Search API (FastAPI)")
        
        uvicorn.run(
            "search.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            app_dir=SRC_DIR
        )

if __name__ == "__main__":
    main()