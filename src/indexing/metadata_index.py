# src/indexing/metadata_index.py
import json
import os
import config

metadata_stores = {}

def load_metadata_from_disk():
    global metadata_stores
    metadata_stores = {}
    if os.path.exists(config.METADATA_DIR):
        for file in os.listdir(config.METADATA_DIR):
            if file.endswith(".json"):
                domain = file.replace(".json", "")
                with open(config.METADATA_DIR / file, "r", encoding="utf-8") as f:
                    metadata_stores[domain] = json.load(f)
        total = sum(len(v) for v in metadata_stores.values())
        print(f"📖 {total} documents chargés depuis {len(metadata_stores)} fichiers de métadonnées.")
    else:
        metadata_stores = {}
        
def store_metadata(entry, domain):
    """Enregistre l'entrée dans le bon domaine et renvoie l'ID local à ce domaine."""
    if domain not in metadata_stores:
        metadata_stores[domain] = []
    
    # L'ID est l'index dans la liste du domaine
    doc_id = len(metadata_stores[domain]) 
    entry["id"] = doc_id
    entry["domain"] = domain
    
    metadata_stores[domain].append(entry)
    return doc_id 

def save_metadata_to_disk():
    os.makedirs(config.METADATA_DIR, exist_ok=True)
    for domain, entries in metadata_stores.items():
        file_path = config.METADATA_DIR / f"{domain}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=4, ensure_ascii=False)
    print(f"Métadonnées sauvegardées par domaine dans : {config.METADATA_DIR}")

def clear_metadata():
    global metadata_stores
    metadata_stores = {}
    if os.path.exists(config.METADATA_DIR):
        import shutil
        shutil.rmtree(config.METADATA_DIR)
        config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        print("Dossier de métadonnées réinitialisé.")

def get_all_metadata():
    all_meta = []
    for entries in metadata_stores.values():
        all_meta.extend(entries)
    return all_meta

def get_metadata_by_id(doc_id, domain):
    try:
        return metadata_stores[domain][doc_id]
    except (KeyError, IndexError):
        return None