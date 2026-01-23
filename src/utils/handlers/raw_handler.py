# src/utils/handlers/raw_handler.py
import os
import re
import numpy as np
from src import config
from collections import Counter
from src.embeddings.image_embeddings import embed_image

def detect_label(filepath=None, text=None, image=None, label_mapping=None, suggested_label=None):
    """Détecteur universel (Image/PDF/TXT) incluant la Couche 0."""
    
    # --- COUCHE 0 : Suggestion Directe ---
    if suggested_label:
        return str(suggested_label).lower().strip()

    label = None
    
    # 1. Dossier Parent
    if filepath:
        dirname = os.path.basename(os.path.dirname(filepath)).lower()
        if label_mapping and dirname in label_mapping:
            return dirname
        
    # 2. Recherche par mot-clé dans le Texte (OCR)
    if text and label_mapping:
        text_lower = text.lower()
        for lbl in sorted(label_mapping.keys(), key=len, reverse=True):
            if lbl in text_lower: return lbl
        
    # 3. CLIP Visuel (Calcul matriciel rapide)
    if image and label_mapping:
        try:
            img_emb = embed_image(image)
            labels_list = list(label_mapping.keys())
            vectors_matrix = np.array(list(label_mapping.values()))
            sims = np.dot(vectors_matrix, img_emb)
            idx_max = np.argmax(sims)
            if sims[idx_max] > config.SEMANTIC_THRESHOLD:
                label = labels_list[idx_max]
        except: pass

    # 4. Fallback statistique
    if not label and text and config.ENABLE_STATISTICAL_FALLBACK:
        words = re.findall(r'\b\w{' + str(config.LABEL_MIN_LENGTH) + r',}\b', text.lower())
        if words: label = f"auto_{Counter(words).most_common(1)[0][0]}"
            
    return label or "unknown"