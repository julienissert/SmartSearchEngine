# src/ingestion/main.py
import time
import argparse
import sys
from pathlib import Path

# Ajout du chemin parent au sys.path pour permettre les imports relatifs si exécuté directement
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from ingestion.service import IngestionService
from utils.logger import setup_logger

logger = setup_logger("IngestionCLI")

def main():
    # 1. Configuration de l'argumentaire CLI
    parser = argparse.ArgumentParser(description="SmartSearchEngine - Pipeline d'ingestion massive")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=['r', 'c', 'R', 'C'], 
        default=None,
        help="Mode d'ingestion : (r)éinitialiser l'index ou (c)ompléter l'existant."
    )
    
    args = parser.parse_args()
    start_time = time.time()

    # 2. Logique de détermination du mode
    # Si le mode n'est pas fourni en argument, on vérifie si une base existe
    mode = args.mode.lower() if args.mode else None

    if mode is None:
        if config.METADATA_DIR.exists() and any(config.METADATA_DIR.iterdir()):
            print(f"\n[AVERTISSEMENT] Base existante détectée dans : {config.METADATA_DIR}")
            choice = input("Voulez-vous (R)éinitialiser ou (C)ompléter la base ? [R/C] : ").lower()
            mode = choice if choice in ['r', 'c'] else 'r'
        else:
            mode = 'r' # Par défaut si rien n'existe

    logger.info(f"Démarrage de l'ingestion - Mode : {'Compléter' if mode == 'c' else 'Réinitialiser'}")

    # 3. Exécution du workflow via le Service (Multiprocessing activé dans IngestionService)
    try:
        # On instancie le service
        service = IngestionService()
        # On lance le workflow (assure-toi que run_workflow accepte le mode en paramètre)
        new_docs, total_files = service.run_workflow(mode)
        
        # 4. Affichage du résumé final
        duration = time.time() - start_time
        print("\n" + "="*40)
        print("📊 INGESTION TERMINÉE")
        print(f"Fichiers analysés     : {total_files}")
        print(f"Documents indexés     : {new_docs}")
        print(f"Temps total           : {duration:.2f} secondes")
        
        if new_docs > 0:
            print(f"Vitesse moyenne       : {duration/new_docs:.4f} s/doc")
        
        logger.info(f"Fin de mission. {new_docs} documents traités en {duration:.2f}s.")
        print("="*40)

    except FileNotFoundError as e:
        logger.error(f"Fichier ou dossier introuvable : {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur fatale lors de l'ingestion : {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()