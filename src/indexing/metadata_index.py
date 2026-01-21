# src/indexing/metadata_index.py
import sqlite3
import json
import os
import shutil
from pathlib import Path
import config
from utils.logger import setup_logger

logger = setup_logger("MetadataIndex")

def get_db_connection():
    conn = sqlite3.connect(config.METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row 
    return conn

def init_db():
    os.makedirs(config.COMPUTED_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id INTEGER,
            domain TEXT,
            source TEXT,
            file_hash TEXT UNIQUE,  
            type TEXT,
            label TEXT,
            domain_score REAL,
            raw_data TEXT,
            snippet TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP -- Date auto
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_label ON metadata (label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_retrieval ON metadata (domain, local_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON metadata (source)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON metadata (file_hash)')
    
    conn.commit()
    conn.close()

def load_metadata_from_disk():
    init_db()

def store_metadata(entry, domain):
    """
    Enregistre les métadonnées en base SQLite.
    Optimisé pour l'ingestion massive : pas de init_db répétitif.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM metadata WHERE domain = ?", (domain,))
    local_id = cursor.fetchone()[0]
    raw_data_str = json.dumps(entry["raw_data"], ensure_ascii=False) if entry.get("raw_data") else None

    try:
        cursor.execute('''
            INSERT INTO metadata (
                local_id, domain, source, file_hash, type, 
                label, domain_score, raw_data, snippet
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            local_id,
            domain,
            entry.get("source", ""),
            entry.get("file_hash", ""), 
            entry.get("type", "unknown"),
            entry.get("label", "unknown"),
            entry.get("domain_score", 0.0),
            raw_data_str,
            entry.get("snippet", "")
        ))
        conn.commit()
        return local_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_metadata_by_id(doc_id, domain):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metadata WHERE local_id = ? AND domain = ?", (doc_id, domain))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        data = dict(row)
        if data["raw_data"]:
            data["raw_data"] = json.loads(data["raw_data"])
        return data
    return None

def get_all_metadata():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT source, file_hash FROM metadata")
    rows = cursor.fetchall()
    conn.close()
    return [{"source": row["source"], "file_hash": row["file_hash"]} for row in rows]

def clear_metadata():
    db_path = config.METADATA_DB_PATH
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.info("Base de données SQLite supprimée.")
        except PermissionError:
            logger.warning("Impossible de supprimer la DB (fichier verrouillé).")
            
    if os.path.exists(config.METADATA_DIR) and config.METADATA_DIR.is_dir():
         shutil.rmtree(config.METADATA_DIR, ignore_errors=True)
         
    init_db()