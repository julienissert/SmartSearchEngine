# src/indexing/faiss_index.py
import faiss
import numpy as np
import os
import pickle
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("FaissIndex")

# Stockage global des index et compteurs par domaine
_indexes = {}
_id_counters = {}

def get_index(domain):
    """Récupère ou crée l'index HNSW pour un domaine donné."""
    global _indexes, _id_counters
    if domain not in _indexes:
        # Configuration HNSW depuis config.py pour l'échelle 80 Go
        M = config.FAISS_HNSW_M
        index = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, M)
        index.hnsw.efConstruction = config.FAISS_HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = config.FAISS_HNSW_EF_SEARCH
        
        _indexes[domain] = index
        # Initialise le compteur si pas chargé depuis le disque
        if domain not in _id_counters:
            _id_counters[domain] = 0
            
        logger.info(f"Index HNSW [{domain}] initialisé (M={M}, efC={index.hnsw.efConstruction})")
    return _indexes[domain]

def add_to_index(vector, domain):
    """
    Ajoute un vecteur et retourne son local_id (sa position exacte).
    """
    if domain == "unknown" or vector is None:
        return -1

    index = get_index(domain)
    
    # ÉLITE : Normalisation L2 impérative pour la précision CLIP
    v = vector.astype('float32').reshape(1, -1)
    faiss.normalize_L2(v)
    
    local_id = index.ntotal
    index.add(v)
    
    # Mise à jour du compteur pour la persistance
    _id_counters[domain] = index.ntotal
    
    return int(local_id)

def save_all_indexes():
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    for domain, index in _indexes.items():
        path = config.FAISS_INDEX_DIR / f"{domain}.index"
        faiss.write_index(index, str(path))
    
    # Sauvegarde des compteurs pour conserver la cohérence après reboot
    with open(config.FAISS_INDEX_DIR / "counters.pkl", "wb") as f:
        pickle.dump(_id_counters, f)
    logger.info("Tous les index et compteurs FAISS sauvegardés.")

def load_all_indexes():
    global _indexes, _id_counters
    if not config.FAISS_INDEX_DIR.exists(): return

    # 1. Chargement des compteurs
    counter_path = config.FAISS_INDEX_DIR / "counters.pkl"
    if counter_path.exists():
        with open(counter_path, "rb") as f:
            _id_counters = pickle.load(f)

    # 2. Chargement des fichiers .index
    for f_name in os.listdir(config.FAISS_INDEX_DIR):
        if f_name.endswith(".index"):
            domain = f_name.replace(".index", "")
            path = config.FAISS_INDEX_DIR / f_name
            _indexes[domain] = faiss.read_index(str(path))
            _indexes[domain].hnsw.efSearch = config.FAISS_HNSW_EF_SEARCH
    logger.info(f"Index chargés : {list(_indexes.keys())}")

def reset_all_indexes():
    """Réinitialisation totale RAM + Disque."""
    global _indexes, _id_counters
    _indexes = {}
    _id_counters = {}
    
    import shutil
    if config.FAISS_INDEX_DIR.exists():
        shutil.rmtree(config.FAISS_INDEX_DIR)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    logger.info("Tous les index FAISS réinitialisés.")