# src/utils/handlers/raw_handler.py
import os
import re
import numpy as np
from src import config
from collections import Counter
from src.embeddings.image_embeddings import embed_image

def resolve_raw_label(filepath=None, text=None, image=None, label_mapping=None, suggested_label=None):
    """Détecteur universel amélioré pour la structure Kaggle archive(n)."""
    
    # --- COUCHE 0 : Suggestion Directe ---
    if suggested_label:
        return str(suggested_label).lower().strip()

    # --- AMÉLIORATION 1 : Analyse du Nom de Fichier (Priorité Haute) ---
    if filepath:
        # Extraire 'prescription_01' de 'archive(1)/prescription_01.pdf'
        filename = os.path.splitext(os.path.basename(filepath))[0].lower()
        # Nettoyage chirurgical
        clean_name = filename.replace("dataset", "").replace("archive", "").replace("real", "").strip("_ ")
        
        if label_mapping and clean_name in label_mapping:
            return clean_name # On a trouvé le label dans le nom du fichier !

    # --- AMÉLIORATION 2 : Dossier Parent (avec filtre anti-générique) ---
    if filepath:
        dirname = os.path.basename(os.path.dirname(filepath)).lower()
        # On ignore les noms de dossiers inutiles type 'archive(1)' ou 'raw-datasets'
        if label_mapping and dirname in label_mapping:
            if not (dirname.startswith("archive") or dirname == "raw-datasets"):
                return dirname
        
    # --- COUCHE 2 : Recherche par mot-clé dans le Texte (OCR) ---
    if text and label_mapping:
        text_lower = text.lower()
        # Tri par longueur pour trouver 'assurance maladie' avant 'maladie'
        for lbl in sorted(label_mapping.keys(), key=len, reverse=True):
            if lbl in text_lower: return lbl
        
    # --- COUCHE 3 : CLIP Visuel (Calcul matriciel rapide) ---
    if image and label_mapping:
        try:
            img_emb = embed_image(image)
            labels_list = list(label_mapping.keys())
            vectors_matrix = np.array(list(label_mapping.values()))
            sims = np.dot(vectors_matrix, img_emb)
            idx_max = np.argmax(sims)
            if sims[idx_max] > config.SEMANTIC_THRESHOLD:
                return labels_list[idx_max]
        except: pass

    # --- COUCHE 4 : Fallback statistique ---
    if text and config.ENABLE_STATISTICAL_FALLBACK:
        words = re.findall(r'\b\w{' + str(config.LABEL_MIN_LENGTH) + r',}\b', text.lower())
        if words: 
            return f"auto_{Counter(words).most_common(1)[0][0]}"
            
    return "unknown"