#utils/preprocessing.py
import re
import os
import hashlib
from pathlib import Path

def calculate_fast_hash(filepath):
    try:
        stats = os.stat(filepath)
        file_size = stats.st_size
        mtime = stats.st_mtime
        
        hasher = hashlib.md5(f"{file_size}_{mtime}".encode())
        
        # Pour les fichiers > 1Mo, on lit juste le début, le milieu et la fin
        if file_size > 1024 * 1024:
            with open(filepath, 'rb') as f:
                hasher.update(f.read(1024 * 10)) # 10 Ko début
                f.seek(file_size // 2)
                hasher.update(f.read(1024 * 10)) # 10 Ko milieu
                f.seek(-min(file_size, 1024 * 10), 2)
                hasher.update(f.read(1024 * 10)) # 10 Ko fin
        else:
            # Petit fichier : on lit tout
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
        return hasher.hexdigest()
    except Exception:
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9éèêàâùç\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compute_text_match_ratio(query: str, target_text: str) -> float:
    """
    Calcule mathématiquement le ratio de mots de la requête présents dans la cible.
    Outil générique réutilisable partout.
    """
    if not query or not target_text:
        return 0.0
    
    q_words = set(clean_text(query).split())
    if not q_words: return 0.0
    
    t_clean = clean_text(target_text)
    matches = sum(1 for word in q_words if word in t_clean)
    
    return min(1.0, matches / len(q_words))

def calculate_folder_signature(folder_path):
    try:
        p = Path(folder_path)
        files = [f for f in p.rglob('*') if f.is_file()]
        if not files: return None
        
        count = len(files)
        total_size = sum(f.stat().st_size for f in files)
        last_mod = max(f.stat().st_mtime for f in files)
        
        return hashlib.md5(f"{folder_path}_{count}_{total_size}_{last_mod}".encode()).hexdigest()
    except Exception:
        return None