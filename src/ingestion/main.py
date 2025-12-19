# src/ingestion/main.py
import os
import time
import config
from ingestion.folder_scanner import scan_folder
from ingestion.dispatcher import dispatch_loader
from ingestion.core import process_document  
from indexing.faiss_index import reset_all_indexes
from indexing.metadata_index import save_metadata_to_disk, clear_metadata
from utils.label_detector import analyze_dataset_structure

def main():
    start_time = time.time()
    
    # 1. Initialisation (Clean Start)
    reset_all_indexes()
    clear_metadata()

    if not os.path.exists(config.DATASET_DIR):
        print(f"Erreur: Dossier {config.DATASET_DIR} introuvable.")
        return

    # 2. Préparation des connaissances
    print("Analyse de la structure du dataset...")
    valid_labels = analyze_dataset_structure(config.DATASET_DIR)
    
    # 3. Scan et Traitement
    print("Début de l'ingestion des fichiers...")
    files = scan_folder(config.DATASET_DIR)
    total_docs = 0 

    for f in files:
        try:
            for doc in dispatch_loader(f):
                process_document(doc, valid_labels)
                total_docs += 1
        except Exception as e:
            print(f"⚠️ Erreur sur {f}: {e}")

    # 4. Finalisation et Persistance
    save_metadata_to_disk()
    
    duration = time.time() - start_time
    print("\n" + "="*40)
    print(f"INGESTION TERMINÉE")
    print(f"Documents traités : {total_docs}")
    print(f"Temps total      : {duration:.2f} secondes")
    if total_docs > 0:
        print(f"Vitesse moyenne  : {duration/total_docs:.4f} s/doc")
    print("="*40)

if __name__ == "__main__":
    main()