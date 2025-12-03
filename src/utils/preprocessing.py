#utils/preprocessing.py
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9éèêàâùç\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
