# src/indexing/faiss_index.py
import faiss
import numpy as np
import os
import pickle
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("FaissIndex")

# Stockage global des index par domaine
_indexes = {}
_id_counters = {}

def get_index(domain):
    """Récupère ou crée l'index FAISS pour un domaine donné."""
    if domain not in _indexes:
        # HNSW pour la vitesse de recherche approximative
        M, ef_c, ef_s = config.FAISS_HNSW_M, config.FAISS_HNSW_EF_CONSTRUCTION, config.FAISS_HNSW_EF_SEARCH
        index = faiss.IndexHNSWFlat(config.EMBEDDING_DIM, M)
        index.hnsw.efConstruction = ef_c
        index.hnsw.efSearch = ef_s
        
        _indexes[domain] = index
        _id_counters[domain] = 0
    return _indexes[domain]

def add_to_index(vector, domain):
    """
    Ajoute un vecteur à l'index et retourne son ID unique.
    CORRECTION : Retourne un int natif, pas un numpy type.
    """
    if domain == "unknown":
        return -1

    index = get_index(domain)
    
    # Normalisation L2 (nécessaire pour similarité cosinus avec FAISS)
    vector = vector.astype('float32')
    faiss.normalize_L2(vector.reshape(1, -1))
    
    # Ajout à l'index
    index.add(vector.reshape(1, -1))
    
    # Gestion de l'ID
    doc_id = _id_counters[domain]
    _id_counters[domain] += 1
    
    # On combine le domaine et l'ID pour avoir une clé unique globale si nécessaire
    # Mais ici on retourne l'ID local simple pour le stockage metadata
    # (Il faudra gérer le mapping ID <-> Domaine si on veut retrouver le doc)
    
    # ASTUCE : Pour l'unicité globale dans SQLite, on peut encoder le domaine dans l'ID
    # Exemple : ID = (hash(domain) << 32) | doc_id
    # Pour l'instant, restons simple : on retourne un grand entier unique
    # On utilise un préfixe basé sur le domaine pour éviter les collisions dans la DB unique
    domain_prefix = abs(hash(domain)) % 1000000
    global_id = int(f"{domain_prefix}{doc_id}")
    
    return int(global_id) # Force le type int Python

def save_all_indexes():
    """Sauvegarde tous les index sur disque."""
    for domain, index in _indexes.items():
        path = config.FAISS_INDEX_DIR / f"{domain}.index"
        faiss.write_index(index, str(path))
    
    # Sauvegarde des compteurs
    with open(config.FAISS_INDEX_DIR / "counters.pkl", "wb") as f:
        pickle.dump(_id_counters, f)

def load_all_indexes():
    """Charge les index depuis le disque."""
    global _indexes, _id_counters
    if not config.FAISS_INDEX_DIR.exists(): return

    # Chargement compteurs
    counter_path = config.FAISS_INDEX_DIR / "counters.pkl"
    if counter_path.exists():
        with open(counter_path, "rb") as f:
            _id_counters = pickle.load(f)

    # Chargement index
    for f in os.listdir(config.FAISS_INDEX_DIR):
        if f.endswith(".index"):
            domain = f.replace(".index", "")
            _indexes[domain] = faiss.read_index(str(config.FAISS_INDEX_DIR / f))

def reset_all_indexes():
    """Vide la mémoire et supprime les fichiers."""
    global _indexes, _id_counters
    _indexes = {}
    _id_counters = {}
    
    import shutil
    if config.FAISS_INDEX_DIR.exists():
        shutil.rmtree(config.FAISS_INDEX_DIR)
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)