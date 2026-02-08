# src/intelligence/handlers/raw_handler.py
import os
import re
import numpy as np
from src import config
from collections import Counter
from src.embeddings.image_embeddings import embed_image
from src.intelligence.handlers.structured_handler import is_label_noisy

def resolve_raw_label(filepath=None, text=None, image=None, image_vector=None, label_mapping=None, suggested_label=None):
    """
    Détecteur astucieux : Arbitre entre Dossier (Classe) et Fichier (Nom).
    Priorise la structure physique pour une vitesse maximale.
    """
    
    # --- COUCHE 0 : Suggestion Directe ---
    if suggested_label:
        return str(suggested_label).lower().strip()

    # --- COUCHE 1 : ARBITRAGE STRUCTUREL (Arbitrage Dossier vs Fichier) ---
    if filepath:
        fname_raw = os.path.basename(filepath)
        # Nettoyage du nom de fichier (on enlève les extensions)
        fname_no_ext = fname_raw.lower()
        while '.' in fname_no_ext:
            fname_no_ext = os.path.splitext(fname_no_ext)[0]
            
        # Récupération du dossier parent immédiat
        parent_folder = os.path.basename(os.path.dirname(filepath)).lower()
        
        # Liste étendue des dossiers "bruit" qui ne sont jamais des labels
        generic_dirs = config.TECHNICAL_FOLDERS + [
            "images", "data", "raw-datasets", "content", "img", 
            "photos", "val", "train", "test", "dataset", "archive"
        ]
        
        # Analyse du bruit
        file_is_noise = is_label_noisy(fname_no_ext)
        folder_is_noise = is_label_noisy(parent_folder) or parent_folder in generic_dirs

        # LOGIQUE D'ARBITRAGE :
        
        # 1. Dossier propre (ex: apple_pie) et fichier bruité (ex: img123.jpg) -> Dossier
        if file_is_noise and not folder_is_noise:
            return parent_folder

        # 2. Dossier bruité (ex: images) et fichier propre (ex: ferrari_enzo.jpg) -> Fichier
        if not file_is_noise and folder_is_noise:
            return fname_no_ext

        # 3. Si les deux sont propres, on privilégie le dossier 
        # (Dans 90% des datasets de classification, le dossier est la classe)
        if not folder_is_noise:
            # Sécurité : si le nom est dans le mapping, on le prend direct
            return parent_folder

        # 4. Si le fichier est propre (Ferrari) mais qu'on n'est pas sûr du dossier, on prend le fichier
        if not file_is_noise:
            return fname_no_ext

    # --- COUCHE 2 : RECHERCHE PAR MOTS-CLÉS DANS LE TEXTE (OCR) ---
    # On n'arrive ici que si la structure est 100% bruitée (ex: images/img123.jpg)
    if text and label_mapping:
        # On extrait les labels si mapping est un dict de vecteurs
        keys = label_mapping.keys() if isinstance(label_mapping, dict) else label_mapping
        text_lower = text.lower()
        for lbl in sorted(keys, key=len, reverse=True):
            if lbl in text_lower: return lbl
        
    # --- COUCHE 3 : CLIP VISUEL (Similitude sémantique) ---
    # Dernier recours : calcul mathématique de l'image
    if (image or image_vector is not None) and label_mapping:
        try:
            img_emb = image_vector if image_vector is not None else embed_image(image)
            
            # Si mapping est un dict de vecteurs
            if isinstance(label_mapping, dict) and len(label_mapping) > 0:
                labels_list = list(label_mapping.keys())
                vectors_matrix = np.array(list(label_mapping.values()))
                
                sims = np.dot(vectors_matrix, img_emb)
                best_idx = np.argmax(sims)

                if sims[best_idx] > config.SEMANTIC_THRESHOLD:
                    return labels_list[best_idx]
                
        except Exception:
            pass

    # --- COUCHE 4 : FALLBACK STATISTIQUE ---
    if text and config.ENABLE_STATISTICAL_FALLBACK:
        words = re.findall(r'\b\w{' + str(config.LABEL_MIN_LENGTH) + r',}\b', text.lower())
        if words: 
            return f"auto_{Counter(words).most_common(1)[0][0]}"
            
    return "unknown"