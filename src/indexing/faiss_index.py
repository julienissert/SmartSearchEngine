# src/indexing/faiss_index.py
import os
import shutil  
import faiss
import numpy as np
import config
from utils.logger import setup_logger

logger = setup_logger("FaissIndex")

indices = {}

def get_index_path(domain):
    return os.path.join(config.FAISS_INDEX_DIR, f"{domain}.index")

def get_index(domain):
    """Initialise ou récupère un index HNSW pour un domaine."""
    global indices
    if domain not in indices:
        path = get_index_path(domain)
        if os.path.exists(path):
            # 1. Chargement d'un index existant
            index = faiss.read_index(path)
            index.hnsw.efSearch = config.FAISS_HNSW_EF_SEARCH
            indices[domain] = index
            logger.info(f"Index HNSW [{domain}] chargé (efSearch={config.FAISS_HNSW_EF_SEARCH})")
        else:
            # 2. Création d'un nouvel index HNSWFlat
            index = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, config.FAISS_HNSW_M)
            
            # Paramètres de précision/vitesse 
            index.hnsw.efConstruction = config.FAISS_HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = config.FAISS_HNSW_EF_SEARCH
            
            indices[domain] = index
            logger.info(f"Nouvel index HNSW [{domain}] créé (M={config.FAISS_HNSW_M}, efC={config.FAISS_HNSW_EF_CONSTRUCTION})")
            
    return indices[domain]

def load_all_indexes():
    """Charge tous les index existants au démarrage."""
    if not os.path.exists(config.FAISS_INDEX_DIR):    
        return
    for filename in os.listdir(config.FAISS_INDEX_DIR):
        if filename.endswith(".index"):
            domain = filename.replace(".index", "")
            get_index(domain) 

def reset_all_indexes():
    """Vide la RAM et supprime les fichiers sur le disque."""
    global indices
    indices = {} 
    if os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    logger.info("Tous les index FAISS (HNSW) ont été réinitialisés.")

def add_to_index(domain, vector, doc_id=None):
    """Ajoute un vecteur à l'index HNSW."""
    if vector is None:
        return

    index = get_index(domain)
    
    if hasattr(vector, "detach"):
        vector = vector.detach().cpu().numpy()
    
    v = np.array(vector).astype("float32")
    if v.ndim == 1:
        v = v.reshape(1, -1)
    
    index.add(v)

def save_all_indexes():
    """Sauvegarde physique des index vers le disque."""
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    for domain, index in indices.items():
        path = get_index_path(domain)
        faiss.write_index(index, path)
        logger.info(f"Index HNSW [{domain}] sauvegardé.")