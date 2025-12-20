# src/ingestion/main.py
import os
import time
import config
from ingestion.service import IngestionService

def main():
    start_time = time.time()
    
    # 1. Interaction Utilisateur
    mode = 'r'  
    if os.path.exists(config.METADATA_FILE):
        print(f"\nBase existante détectée ({config.METADATA_FILE})")
        choice = input("Voulez-vous (R)éinitialiser ou (C)ompléter la base ? [R/C] : ").lower()
        if choice == 'c':
            mode = 'c'

    print(f"\nLancement de l'ingestion (Mode: {'Compléter' if mode == 'c' else 'Réinitialiser'})...")

    # 2. Exécution du workflow via le Service
    try:
        new_docs, total_files = IngestionService.run_workflow(mode)
        
        # 3. Affichage du résumé final
        duration = time.time() - start_time
        print("\n" + "="*40)
        print("✨ INGESTION TERMINÉE")
        print(f"Fichiers analysés     : {total_files}")
        print(f"Documents indexés     : {new_docs}")
        print(f"Temps total           : {duration:.2f} secondes")
        if new_docs > 0:
            print(f"Vitesse moyenne       : {duration/new_docs:.4f} s/doc")
        print("="*40)

    except FileNotFoundError as e:
        print(f"\nErreur de configuration : {e}")
    except Exception as e:
        print(f"\nUne erreur imprévue est survenue : {e}")

if __name__ == "__main__":
    main()