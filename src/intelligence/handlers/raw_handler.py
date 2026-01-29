# src/utils/handlers/raw_handler.py
import os
import re
import numpy as np
from src import config
from collections import Counter
from src.embeddings.image_embeddings import embed_image

def resolve_raw_label(filepath=None, text=None, image=None, image_vector=None, label_mapping=None, suggested_label=None):
    """Détecteur universel amélioré pour la structure Kaggle archive(n)."""
    
    # --- COUCHE 0 : Suggestion Directe ---
    if suggested_label:
        return str(suggested_label).lower().strip()

# --- COUCHE 1 : Analyse Structurelle (Nom & Escalade de Dossier) ---
    if filepath:
        basename = os.path.basename(filepath).lower()
        while '.' in basename:
            basename = os.path.splitext(basename)[0]
        
        if label_mapping and basename in label_mapping:
            return basename 
        current_dir = os.path.dirname(filepath)
        for _ in range(3): 
            dirname = os.path.basename(current_dir).lower()
            if dirname and dirname not in config.TECHNICAL_FOLDERS:
                if label_mapping and dirname in label_mapping:
                    return dirname
                break 
            current_dir = os.path.dirname(current_dir)
        
    # --- COUCHE 2 : Recherche par mot-clé dans le Texte (OCR) ---
    if text and label_mapping:
        text_lower = text.lower()
        for lbl in sorted(label_mapping.keys(), key=len, reverse=True):
            if lbl in text_lower: return lbl
        
    # --- COUCHE 3 : CLIP Visuel ---
    if (image or image_vector is not None) and label_mapping:
        try:
            img_emb = image_vector if image_vector is not None else embed_image(image)
            
            labels_list = list(label_mapping.keys())
            vectors_matrix = np.array(list(label_mapping.values()))
            
            # Calcul de similarité cosinus (produit scalaire sur vecteurs normés)
            sims = np.dot(vectors_matrix, img_emb)
            
            # Transformation en probabilités (Softmax) pour évaluer la certitude
            logits = sims * 10.0 
            probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
            
            # Tri pour identifier les deux meilleurs candidats
            idx_sorted = np.argsort(probs)[::-1]
            best_idx = idx_sorted[0]
            
            # GAP ANALYSIS : Arbitrage de sécurité
            if len(idx_sorted) > 1:
                margin = probs[best_idx] - probs[idx_sorted[1]]
                if margin < 0.15: 
                    return "unknown" 

            if sims[best_idx] > config.SEMANTIC_THRESHOLD:
                return labels_list[best_idx]
                
        except Exception as e:
            pass

    # --- COUCHE 4 : Fallback statistique ---
    if text and config.ENABLE_STATISTICAL_FALLBACK:
        words = re.findall(r'\b\w{' + str(config.LABEL_MIN_LENGTH) + r',}\b', text.lower())
        if words: 
            return f"auto_{Counter(words).most_common(1)[0][0]}"
            
    return "unknown"