# src/indexing/metadata_index.py
import sqlite3
import json
import os
import shutil
from pathlib import Path
import config
from src.utils.logger import setup_logger

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
            local_id INTEGER,    -- L'ID utilisé par FAISS (0, 1, 2... par domaine)
            domain TEXT,        -- 'food', 'medical', etc.
            source TEXT,        -- Chemin du fichier
            type TEXT,          -- 'image', 'csv', 'pdf'
            label TEXT,         -- Le label principal
            domain_score REAL,
            raw_data TEXT,      -- Contenu structuré complet (dumps JSON)
            snippet TEXT        -- Extrait de texte
        )
    ''')
    

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_label ON metadata (label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_retrieval ON metadata (domain, local_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON metadata (source)')
    
    conn.commit()
    conn.close()


def load_metadata_from_disk():
    init_db()

def store_metadata(entry, domain):

    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM metadata WHERE domain = ?", (domain,))
    local_id = cursor.fetchone()[0]
    
    raw_data_str = None
    if entry.get("raw_data"):
        raw_data_str = json.dumps(entry["raw_data"], ensure_ascii=False)

    cursor.execute('''
        INSERT INTO metadata (local_id, domain, source, type, label, domain_score, raw_data, snippet)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        local_id,
        domain,
        entry.get("source", ""),
        entry.get("type", "unknown"),
        entry.get("label", "unknown"),
        entry.get("domain_score", 0.0),
        raw_data_str,
        entry.get("snippet", "")
    ))
    
    conn.commit()
    conn.close()
    
    return local_id 

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
    cursor.execute("SELECT source FROM metadata")
    rows = cursor.fetchall()
    conn.close()
    return [{"source": row["source"]} for row in rows]

def save_metadata_to_disk():
    """
    DEPRECATED mais conservé pour compatibilité.
    SQLite sauvegarde à chaque insertion (Autocommit), donc cette fonction ne fait rien.
    Pour rester compatible avec service.py qui l'appelle systématiquement. 
    SUPPRESSION PROCHAIN COMMIT
    """
    pass

def clear_metadata():
    """Supprime physiquement le fichier de base de données."""
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