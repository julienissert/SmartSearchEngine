# src/indexing/metadata_index.py
import sqlite3
import json
import os
import shutil
from src import config
from src.utils.logger import setup_logger

logger = setup_logger("MetadataIndex")

def get_db_connection():
    conn = sqlite3.connect(config.METADATA_DB_PATH)
    conn.row_factory = sqlite3.Row 
    conn.execute("PRAGMA journal_mode=WAL") 
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA cache_size={config.DYNAMIC_CACHE_SIZE}")
    return conn

def init_db():
    os.makedirs(config.COMPUTED_DIR, exist_ok=True)
    conn = get_db_connection() 
    cursor = conn.cursor()
     
    # Table principale : L'ID est lié à FAISS, le Hash est unique
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_id INTEGER,          -- L'ID renvoyé par FAISS pour ce domaine
            domain TEXT,               -- medical, food, etc.
            source TEXT,               -- Chemin complet du fichier
            file_hash TEXT UNIQUE,     -- Empreinte unique (Anti-doublon)
            type TEXT,                 -- Extension (pdf, img, csv)
            label TEXT,                -- Label sémantique (Dossier ou LLM)
            domain_score REAL,         -- Score de confiance CLIP/Detector
            content TEXT,              -- Contexte riche (jusqu'à 20k chars) pour le LLM
            snippet TEXT,              -- Résumé court (500 chars) pour l'UI rapide
            extra TEXT,        -- Données brutes et bonus (OCR, etc.)
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Table des Contrats (Flasher les domaines par dossier)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS folder_contracts (
            folder_path TEXT PRIMARY KEY,
            assigned_domain TEXT,
            confidence REAL,
            is_verified INTEGER DEFAULT 0 -- 1 si validé par LLM/Humain
        )
    ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_retrieval ON metadata (domain, local_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON metadata (file_hash)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_label ON metadata (label)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON metadata (source)')
    
    conn.commit()
    conn.close()

def store_metadata_batch(metadata_list):
    """Insertion par lots (Batch) pour une vitesse d'écriture maximale."""
    if not metadata_list: return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        data_to_insert = [
            (
                m.get('local_id'),
                m.get('domain'),
                str(m.get('source')),
                m.get('file_hash'),
                m.get('type', 'unknown'),
                m.get('label', 'unknown'),
                m.get('domain_score', 0.0),
                m.get('content', '')[:20000], 
                m.get('snippet') or m.get('content', '')[:500],   
                json.dumps(m.get('extra', {}), ensure_ascii=False) 
            )
            for m in metadata_list
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO metadata (
                local_id, domain, source, file_hash, type, 
                label, domain_score, content, snippet, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion batch SQL : {e}")
        conn.rollback()
    finally:
        conn.close()

# --- LOGIQUE DES CONTRATS DE DOSSIER ---
def get_folder_contract(path_to_check):
    """Vérifie si ce chemin (dossier archive) a un contrat."""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT assigned_domain FROM folder_contracts WHERE folder_path = ?", 
        (str(path_to_check),)
    ).fetchone()
    conn.close()
    return row['assigned_domain'] if row else None

def save_folder_contract(folder_path, domain, confidence=1.0, verified=0):
    """Enregistre un contrat de domaine pour un dossier complet."""
    conn = get_db_connection()
    conn.execute('''
        INSERT OR REPLACE INTO folder_contracts (folder_path, assigned_domain, confidence, is_verified)
        VALUES (?, ?, ?, ?)
    ''', (folder_path, domain, confidence, verified))
    conn.commit()
    conn.close()

# --- UTILITAIRES DE MAINTENANCE ---

def check_file_status(file_hash, source_path):
    conn = get_db_connection()
    row = conn.execute("SELECT source FROM metadata WHERE file_hash = ?", (file_hash,)).fetchone()
    conn.close()
    if not row: return 'new'
    return 'exists' if row['source'] == source_path else 'moved'

def update_file_source(file_hash, new_source):
    conn = get_db_connection()
    conn.execute("UPDATE metadata SET source = ? WHERE file_hash = ?", (new_source, file_hash))
    conn.commit()
    conn.close()

def clear_metadata():
    db_path = config.METADATA_DB_PATH
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except Exception: pass
    init_db()
    logger.info("Base de données SQL réinitialisée proprement.")

def load_metadata_from_disk():
    init_db()

def get_all_metadata():
    conn = get_db_connection()
    rows = conn.execute("SELECT source, file_hash FROM metadata").fetchall()
    conn.close()
    return rows