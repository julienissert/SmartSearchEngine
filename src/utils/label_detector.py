# src/utils/label_detector.py
import os
import re
import numpy as np
import config
from collections import Counter
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

_STRUCTURED_FILE_CACHE = {}

def analyze_dataset_structure(dataset_path):
    """Apprend les labels et pré-calcule leurs vecteurs pour une détection rapide."""
    valid_labels = set()
    leaf_folders = []
    
    print(f"Analyse du dataset : {dataset_path}...")

    for root, dirs, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
        for f in files:
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as txt:
                        lines = [l.strip().lower() for l in txt.readlines() 
                                 if len(l.strip()) >= config.LABEL_MIN_LENGTH]
                        valid_labels.update(lines)
                except: pass

    blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset"]
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            if name.lower() not in blacklist and (count/total < 0.15):
                valid_labels.add(name.lower())

    print(f" Génération des vecteurs pour {len(valid_labels)} labels...")
    label_mapping = {lbl: embed_text(lbl) for lbl in valid_labels if lbl}
    return label_mapping

def resolve_structured_label(data_dict, source_path, label_mapping=None, suggested_label=None, dataset_name=None):
    """Stratégie hybride incluant la Couche 0 (Suggestion directe)."""
    
    # --- COUCHE 0 : Validation Immédiate (ex: Attributs H5 / Métadonnées Système) ---
    if suggested_label:
        return str(suggested_label).lower().strip()

    # Cache composite : évite les conflits entre datasets d'un même fichier (H5)
    cache_key = f"{source_path}::{dataset_name}" if dataset_name else source_path
    
    if cache_key in _STRUCTURED_FILE_CACHE:
        target_key = _STRUCTURED_FILE_CACHE[cache_key]
        return str(data_dict.get(target_key, "unknown")).lower().strip()

    keys = list(data_dict.keys())
    
    # --- COUCHE 1 : Correspondance exacte avec le vocabulaire ---
    if label_mapping:
        for key, value in data_dict.items():
            val_clean = str(value).lower().strip()
            if val_clean in label_mapping:
                _STRUCTURED_FILE_CACHE[cache_key] = key
                return val_clean

    # --- COUCHE 2 : CLIP Sémantique (Analyse des en-têtes) ---
    target_concepts = ["product name", "item label", "object name", "category"]
    concept_embs = [embed_text(c) for c in target_concepts]
    ref_vector = np.mean(concept_embs, axis=0)

    best_key, best_score = None, -1.0
    for key in keys:
        key_emb = embed_text(str(key).lower())
        score = np.dot(key_emb, ref_vector)
        if score > best_score:
            best_score, best_key = score, key
    
    if best_score > config.SEMANTIC_THRESHOLD:
        _STRUCTURED_FILE_CACHE[cache_key] = best_key
        return str(data_dict.get(best_key)).lower().strip()

    # --- COUCHE 3 : Profilage Statistique ---
    for key, value in data_dict.items():
        val_str = str(value).strip()
        if config.LABEL_MIN_LENGTH < len(val_str) < config.LABEL_MAX_LENGTH and not val_str.isdigit():
            _STRUCTURED_FILE_CACHE[cache_key] = key
            return val_str.lower()

    return "unknown"

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