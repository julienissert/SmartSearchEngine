# src/indexing/vector_store.py
import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
import json
import os
from src import config
from src.utils.logger import setup_logger
import time 
import random

logger = setup_logger("VectorStore")

MAX_RETRIES = 5
BASE_DELAY = 0.2  

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
        pa.field("signature", pa.string()),
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
    """Insertion atomique avec normalisation L2 et Retry Sécurité (Windows/Rust)."""
    if not metadata_list or not vector_list:
        return 0
    
    # Configuration du Retry
    MAX_RETRIES = 5
    BASE_DELAY = 0.2 
    
    table = init_tables()
    rows = []

    # --- ÉTAPE 1 : PRÉPARATION ET NORMALISATION ---
    for meta, vec in zip(metadata_list, vector_list):
        # Normalisation L2 (Essentiel pour la précision CLIP)
        v = np.array(vec).astype('float32')
        norm = np.linalg.norm(v)
        v = v / norm if norm > 0 else v
        
        content_str = str(meta.get('content', ''))
        
        rows.append({
            "vector": v.tolist(),
            "source": str(meta.get('source', '')),
            "file_hash": str(meta.get('file_hash', '')),
            "type": str(meta.get('type', 'unknown')),
            "domain": str(meta.get('domain', 'unknown')),
            "label": str(meta.get('label', 'unknown')),
            "domain_score": float(meta.get('domain_score', 0.0)),
            "content": content_str[:20000], 
            "snippet": str(meta.get('snippet') or content_str[:500]),
            "extra": json.dumps(meta.get('extra', {}), ensure_ascii=False)
        })

    # --- ÉTAPE 2 : INSERTION AVEC RETRY (BOUCLE ANTI-COLLISION RUST) ---
    for attempt in range(MAX_RETRIES):
        try:
            table.add(rows)
            return len(rows) # Succès !
            
        except Exception as e:
            # On détecte si c'est une erreur de verrouillage fichier (propre à Windows)
            error_msg = str(e).lower()
            is_lock_error = any(x in error_msg for x in ["access denied", "locked", "being used", "permission denied"])
            
            if is_lock_error and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1)
                logger.warning(f"⚠️ Collision Windows détectée (Attempt {attempt+1}/{MAX_RETRIES}). Retry dans {wait_time:.2f}s...")
                time.sleep(wait_time)
                table = init_tables()
            else:
                logger.error(f"Échec critique insertion LanceDB : {e}")
                return 0
    
    return 0

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
    """Récupère le contrat complet d'un dossier."""
    db = get_db()
    if "folder_contracts" not in list(db.table_names()): return None
    table = db.open_table("folder_contracts")
    res = table.search().where(f"folder_path = '{str(folder_path)}'", prefilter=True).to_pandas()
    return res.iloc[0].to_dict() if not res.empty else None

def save_folder_contract(folder_path, domain, signature,confidence=1.0, verified=0):
    """Enregistre ou met à jour un contrat de dossier (Logique Upsert)."""
    db = get_db()
    table = db.open_table("folder_contracts")
    table.delete(f"folder_path = '{str(folder_path)}'")
    
    table.add([{
        "folder_path": str(folder_path),
        "signature": str(signature),
        "assigned_domain": domain,
        "confidence": float(confidence),
        "is_verified": int(verified)
    }])

def reset_store():
    """Réinitialisation totale (Base de données + Cache schémas)."""
    db = get_db()
    for t in db.table_names():
        db.drop_table(t)
    
    if config.SCHEMA_CACHE_PATH.exists():
        config.SCHEMA_CACHE_PATH.unlink()
        logger.info(f"Mémoire sémantique effacée : {config.SCHEMA_CACHE_PATH.name}")
    
    init_tables()
    logger.info("Store LanceDB totalement réinitialisé (Page blanche).")
    
def get_all_indexed_hashes():
    """Récupère TOUTES les signatures (hashes) pour éviter de ré-ingérer l'existant."""
    try:
        db = get_db()
        all_tables = list(db.table_names())
        if config.TABLE_NAME not in all_tables: 
            return set()
        
        table = db.open_table(config.TABLE_NAME)
        
        if table.count_rows() == 0:
            return set()
        query_builder = table.search() 
        df = query_builder.select(["file_hash"]).to_pandas()
        
        if "file_hash" in df.columns:
            hashes = df["file_hash"].dropna().unique().astype(str).tolist()
            indexed_set = set(hashes)
            logger.info(f"Fast-Check : {len(indexed_set)} signatures uniques trouvées en base.")
            return indexed_set
        
        return set()

    except Exception as e:
        logger.error(f"Erreur Fast-Check (récupération hashes) : {e}")
        return set()

def create_vector_index():
    """Crée un index IVF-PQ pour garantir des recherches sub-secondes sur disque."""
    table = init_tables()
    # On crée l'index uniquement s'il y a assez de données (ex: > 1000 docs)
    if len(table) > 1000:
        logger.info("Construction de l'index vectoriel sur disque...")
        table.create_index(metric="cosine", num_partitions=256, num_sub_vectors=64)
        logger.info(" Index vectoriel optimisé.")