#utils/preprocessing.py
import re
import os
import hashlib

def calculate_fast_hash(filepath):
    try:
        stats = os.stat(filepath)
        file_size = stats.st_size
        mtime = stats.st_mtime
        
        # taille et la date 
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
