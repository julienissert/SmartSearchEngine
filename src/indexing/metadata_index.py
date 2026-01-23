# src/indexing/metadata_index.py
import sqlite3
import json
import os
import config
from utils.logger import setup_logger

logger = setup_logger("MetadataIndex")

def get_db_connection():
    """Crée une connexion optimisée pour l'ingestion massive."""
    conn = sqlite3.connect(config.METADATA_DB_PATH)
    # Optimisations PRAGMA pour la vitesse d'écriture
    conn.execute(f"PRAGMA cache_size = {config.DYNAMIC_CACHE_SIZE}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY,
            source TEXT,
            file_hash TEXT UNIQUE,
            domain TEXT,
            content TEXT,
            extra_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def clear_metadata():
    if os.path.exists(config.METADATA_DB_PATH):
        try:
            os.remove(config.METADATA_DB_PATH)
        except OSError:
            pass 
    init_db()
    logger.info("Base de données SQLite réinitialisée.")

def store_metadata_batch(metadata_list):
    """
    Insère une liste de métadonnées en une seule transaction SQL.
    Crucial pour la performance (Batch Insert).
    """
    if not metadata_list: return
    
    conn = get_db_connection()
    try:
        # Préparation des tuples pour executemany
        data_to_insert = [
            (
                m['id'],
                m.get('source'),
                m.get('file_hash'),
                m.get('domain'),
                m.get('content', ''),
                json.dumps(m.get('extra', {}))
            )
            for m in metadata_list
        ]
        
        conn.executemany('''
            INSERT OR REPLACE INTO metadata (id, source, file_hash, domain, content, extra_json)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur SQL Batch : {e}")
    finally:
        conn.close()

def check_file_status(file_hash, current_path):
    conn = get_db_connection()
    try:
        row = conn.execute('SELECT source FROM metadata WHERE file_hash = ?', (file_hash,)).fetchone()
        if not row: return 'new'
        return 'exists' if row['source'] == str(current_path) else 'moved'
    finally:
        conn.close()

def update_file_source(file_hash, new_path):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE metadata SET source = ? WHERE file_hash = ?', (str(new_path), file_hash))
        conn.commit()
    finally:
        conn.close()

def load_metadata_from_disk():
    init_db()

def get_all_metadata():
    conn = get_db_connection()
    try:
        rows = conn.execute('SELECT * FROM metadata').fetchall()
        return rows
    finally:
        conn.close()