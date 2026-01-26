# src/indexing/vector_store.py
import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
import json
import os
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("VectorStore")

_db_connection = None

def get_db():
    """Singleton de connexion avec gestion de dossier automatique."""
    global _db_connection
    if _db_connection is None:
        config.LANCEDB_URI.mkdir(parents=True, exist_ok=True)
        _db_connection = lancedb.connect(config.LANCEDB_URI)
    return _db_connection

def init_tables():
    """Initialise le catalogue et la table des contrats (Schémas Élite)."""
    db = get_db()
    
    # 1. Schéma de la Table principale (Vecteurs + Métadonnées)
    catalog_schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), config.EMBEDDING_DIM)),
        pa.field("source", pa.string()),
        pa.field("file_hash", pa.string()),
        pa.field("type", pa.string()),
        pa.field("domain", pa.string()),
        pa.field("label", pa.string()),
        pa.field("domain_score", pa.float32()),
        pa.field("content", pa.string()),
        pa.field("snippet", pa.string()),
        pa.field("extra", pa.string())  
    ])

    # 2. Schéma des Contrats de dossier (Sans vecteur, pour la rapidité)
    contract_schema = pa.schema([
        pa.field("folder_path", pa.string()),
        pa.field("assigned_domain", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("is_verified", pa.int32())
    ])

    if config.TABLE_NAME not in db.table_names():
        db.create_table(config.TABLE_NAME, schema=catalog_schema)
    
    if "folder_contracts" not in db.table_names():
        db.create_table("folder_contracts", schema=contract_schema)

    return db.open_table(config.TABLE_NAME)

def add_documents(metadata_list, vector_list):
    """Insertion atomique avec normalisation L2 (Héritage FAISS Elite)."""
    if not metadata_list or not vector_list:
        return 0
    
    table = init_tables()
    rows = []

    for meta, vec in zip(metadata_list, vector_list):
        # --- ROBUSTESSE : Normalisation L2 impérative (Standard CLIP) ---
        v = np.array(vec).astype('float32')
        norm = np.linalg.norm(v)
        v = v / norm if norm > 0 else v
        
        # --- PRÉPARATION : Mapping des colonnes ---
        content_str = meta.get('content', '')
        rows.append({
            "vector": v.tolist(),
            "source": str(meta.get('source')),
            "file_hash": meta.get('file_hash'),
            "type": meta.get('type', 'unknown'),
            "domain": meta.get('domain', 'unknown'),
            "label": meta.get('label', 'unknown'),
            "domain_score": float(meta.get('domain_score', 0.0)),
            "content": content_str[:20000], 
            "snippet": meta.get('snippet') or content_str[:500],
            "extra": json.dumps(meta.get('extra', {}), ensure_ascii=False)
        })

    # LanceDB gère l'atomicité de l'écriture
    table.add(rows)
    return len(rows)

# --- LOGIQUE DE MAINTENANCE (Portage SQLite) ---

def check_file_status(file_hash, source_path):
    """Détecte les nouveaux fichiers, doublons ou déplacements."""
    db = get_db()
    if config.TABLE_NAME not in db.table_names(): return 'new'
    
    table = db.open_table(config.TABLE_NAME)
    res = table.search().where(f"file_hash = '{file_hash}'").to_pandas()
    
    if res.empty: return 'new'
    return 'exists' if res.iloc[0]['source'] == str(source_path) else 'moved'

def update_file_source(file_hash, new_source):
    """Met à jour le chemin d'un fichier déplacé."""
    table = init_tables()
    table.update(where=f"file_hash = '{file_hash}'", values={"source": str(new_source)})

def get_folder_contract(folder_path):
    """Récupère le contrat de domaine d'un dossier."""
    db = get_db()
    if "folder_contracts" not in db.table_names(): return None
    table = db.open_table("folder_contracts")
    res = table.search().where(f"folder_path = '{str(folder_path)}'").to_pandas()
    return res.iloc[0]['assigned_domain'] if not res.empty else None

def save_folder_contract(folder_path, domain, confidence=1.0, verified=0):
    """Enregistre ou met à jour un contrat de dossier (Logique Upsert)."""
    db = get_db()
    table = db.open_table("folder_contracts")
    table.delete(f"folder_path = '{str(folder_path)}'")
    
    table.add([{
        "folder_path": str(folder_path),
        "assigned_domain": domain,
        "confidence": float(confidence),
        "is_verified": int(verified)
    }])

def reset_store():
    """Réinitialisation totale (Nettoyage physique du disque)."""
    db = get_db()
    for t in db.table_names():
        db.drop_table(t)
    init_tables()
    logger.info("Store LanceDB totalement réinitialisé.")
    
def get_all_indexed_hashes():
    """Récupère tous les hashs pour le Fast-Check incremental."""
    db = get_db()
    if config.TABLE_NAME not in db.table_names(): return set()
    
    table = db.open_table(config.TABLE_NAME)
    df = table.search().select(["file_hash"]).to_pandas()
    return set(df["file_hash"].tolist())

def create_vector_index():
    """Crée un index IVF-PQ pour garantir des recherches sub-secondes sur disque."""
    table = init_tables()
    # On crée l'index uniquement s'il y a assez de données (ex: > 1000 docs)
    if len(table) > 1000:
        logger.info("Construction de l'index vectoriel sur disque...")
        table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=64)
        logger.info(" Index vectoriel optimisé.")