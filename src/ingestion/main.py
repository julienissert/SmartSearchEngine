# src/ingestion/main.py
import time
import config
from ingestion.service import IngestionService
from utils.logger import setup_logger

logger = setup_logger("IngestionCLI")

def main():
    start_time = time.time()
    mode = 'r'  
    
    if config.METADATA_DIR.exists() and any(config.METADATA_DIR.iterdir()):
        print(f"\n Base existante détectée dans : {config.METADATA_DIR}")
        choice = input("Voulez-vous (R)éinitialiser ou (C)ompléter la base ? [R/C] : ").lower()
        if choice == 'c':
            mode = 'c'

    print(f"\n Lancement de l'ingestion (Mode: {'Compléter' if mode == 'c' else 'Réinitialiser'})...")

    # 2. Exécution du workflow via le Service
    try:
        new_docs, total_files = IngestionService.run_workflow(mode)
        
        # 3. Affichage du résumé final
        duration = time.time() - start_time
        print("\n" + "="*40)
        print("INGESTION TERMINÉE")
        print(f"Fichiers analysés     : {total_files}")
        print(f"Documents indexés     : {new_docs}")
        print(f"Temps total           : {duration:.2f} secondes")
        if new_docs > 0:
            print(f"Vitesse moyenne       : {duration/new_docs:.4f} s/doc")
        print("="*40)

    except FileNotFoundError as e:
        logger.error(f"Fichier introuvable durant l'ingestion : {e}")
        print(f"\n Erreur de configuration : {e}")
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)
        print(f"\n Une erreur imprévue est survenue : {e}")

if __name__ == "__main__":
    main()