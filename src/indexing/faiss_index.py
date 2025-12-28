# src/indexing/faiss_index.py
import os
import shutil  
import faiss
import numpy as np
import config

indexes = {}

def load_all_indexes():
    """Charge tous les index existants du disque vers la RAM."""
    global indexes
    if not os.path.exists(config.FAISS_INDEX_DIR):    
        return
    
    for filename in os.listdir(config.FAISS_INDEX_DIR):
        if filename.endswith(".index"):
            domain = filename.replace(".index", "")
            path = os.path.join(config.FAISS_INDEX_DIR, filename)
            indexes[domain] = faiss.read_index(path)
            
def get_index_path(domain):
    return os.path.join(config.FAISS_INDEX_DIR, f"{domain}.index")

def reset_all_indexes():
    """Vide la RAM et supprime les fichiers sur le disque."""
    global indexes
    indexes = {} # Réinitialise le cache mémoire
    if os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    print(f"Index FAISS réinitialisés dans : {config.FAISS_INDEX_DIR}")

def add_to_index(domain, vector, doc_id):
    """Ajoute un vecteur en RAM uniquement (Opération quasi instantanée)."""
    # 1. On vérifie si l'index est déjà chargé en RAM
    if domain not in indexes:
        path = get_index_path(domain)
        if os.path.exists(path):
            indexes[domain] = faiss.read_index(path)
        else:
            # Création d'un nouvel index vide en RAM
            base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
            indexes[domain] = faiss.IndexIDMap(base_index)
    
    # 2. Ajout technique dans la structure en mémoire vive
    v = np.array([vector]).astype("float32")
    ids = np.array([doc_id]).astype("int64")
    indexes[domain].add_with_ids(v, ids)

def save_all_indexes():
    """Sauvegarde physique de tous les index de la RAM vers le disque (Une seule fois à la fin)."""
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    for domain, index in indexes.items():
        path = get_index_path(domain)
        faiss.write_index(index, path)
