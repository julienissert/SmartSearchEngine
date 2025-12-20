# src/ingestion/main.py
import os
import time
import config
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_document  
from indexing.faiss_index import reset_all_indexes
from indexing.metadata_index import (
    save_metadata_to_disk, 
    clear_metadata, 
    load_metadata_from_disk, 
    get_all_metadata
)
from utils.label_detector import analyze_dataset_structure

def main():
    start_time = time.time()
    
    # --- LOGIQUE DE CHOIX DU MODE ---
    mode = 'r'  
    if os.path.exists(config.METADATA_FILE):
        print(f"\n Une base de données existante a été trouvée ({config.METADATA_FILE}).")
        choice = input("Voulez-vous (R)éinitialiser ou (C)ompléter la base ? [R/C] : ").lower()
        if choice == 'c':
            mode = 'c'

    if mode == 'r':
        print("Nettoyage complet des anciens index et métadonnées...")
        reset_all_indexes()
        clear_metadata()
    else:
        print("Chargement des données existantes pour complétion...")
        load_metadata_from_disk()

    if not os.path.exists(config.DATASET_DIR):
        print(f"Erreur: Dossier {config.DATASET_DIR} introuvable.")
        return

    # 1. Analyse de la structure
    print("Analyse de la structure du dataset...")
    valid_labels = analyze_dataset_structure(config.DATASET_DIR)
    
    # 2. Scan et Filtrage des doublons
    print("Scan des fichiers...")
    all_files = scan_folder(config.DATASET_DIR)
    
    if mode == 'c':
        # On récupère les sources déjà présentes dans le JSON pour ne pas les ré-ingérer
        processed_sources = {m['source'] for m in get_all_metadata()}
        files_to_process = [f for f in all_files if f not in processed_sources]
        print(f"ℹ️  {len(all_files) - len(files_to_process)} fichiers déjà indexés ont été ignorés.")
    else:
        files_to_process = all_files

    # 3. Traitement des nouveaux fichiers
    print(f"Début de l'ingestion de {len(files_to_process)} fichiers...")
    total_new_docs = 0 

    for f in files_to_process:
        try:
            for doc in dispatch_loader(f):
                process_document(doc, valid_labels)
                total_new_docs += 1
        except Exception as e:
            print(f"Erreur sur {f}: {e}")

    # 4. Finalisation
    if total_new_docs > 0:
        save_metadata_to_disk()
    
    duration = time.time() - start_time
    print("\n" + "="*40)
    print(f"INGESTION TERMINÉE")
    print(f"Nouveaux documents ajoutés : {total_new_docs}")
    print(f"Temps total: {duration:.2f} secondes")
    if total_new_docs > 0:
        print(f"Vitesse moyenne : {duration/total_new_docs:.4f} s/doc")
    print("="*40)

if __name__ == "__main__":
    main()