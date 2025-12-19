# src/indexing/metadata_index.py
import json
import os
import config

metadata_store = []

def store_metadata(entry):
    doc_id = len(metadata_store) 
    entry["id"] = doc_id
    
    metadata_store.append(entry)
    return doc_id 

def save_metadata_to_disk():
    os.makedirs(os.path.dirname(config.METADATA_FILE), exist_ok=True)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=4, ensure_ascii=False)
    print(f"Base de données JSON sauvegardée : {config.METADATA_FILE}")

def clear_metadata():
    global metadata_store
    metadata_store = []
    if os.path.exists(config.METADATA_FILE):
        os.remove(config.METADATA_FILE)
        print("Ancien fichier de métadonnées supprimé.")

def search_by_label(label):
    return [e for e in metadata_store if e.get("label") == label]

def get_all_metadata():
    return metadata_store

def get_metadata_by_id(doc_id):
    try:
        return metadata_store[doc_id]
    except (IndexError, TypeError):
        return None