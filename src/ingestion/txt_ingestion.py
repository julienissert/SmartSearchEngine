# scripts/ingestion/txt_ingestion.py (Correction)
import os
from utils.preprocessing import clean_text

def load_txt(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    
    return [{
        "source": path,
        "type": "txt",
        "content": cleaned
    }]