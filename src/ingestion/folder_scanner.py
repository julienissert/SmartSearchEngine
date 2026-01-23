# src/ingestion/folder_scanner.py
import os
from src.ingestion.dispatcher import get_supported_extensions 

def scan_folder(folder):
    files = []
    valid_extensions = get_supported_extensions()
    
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(root, name))
    return files