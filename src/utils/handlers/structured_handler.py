# src/utils/handlers/structured_handler.py
import numpy as np
import config
from embeddings.text_embeddings import embed_text

# Cache pour mémoriser la colonne cible par fichier/dataset
_STRUCTURED_FILE_CACHE = {}

def resolve_structured_label(data_dict, source_path, label_mapping=None, suggested_label=None, dataset_name=None):
    """Stratégie hybride pour CSV/H5 (Layers 0-3)."""
    
    # --- COUCHE 0 : Validation Immédiate (ex: Attributs H5) ---
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