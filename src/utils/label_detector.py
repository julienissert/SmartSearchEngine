# src/utils/label_detector.py
import os
import config
from collections import Counter
from tqdm import tqdm
from embeddings.text_embeddings import embed_text_batch

# --- AJOUT DES IMPORTS MANQUANTS POUR CORE.PY ---
from .handlers.structured_handler import resolve_structured_label
from .handlers.raw_handler import detect_label

def analyze_dataset_structure(dataset_path):
    """
    Analyse la structure du dataset et pré-calcule les vecteurs CLIP pour tous les labels.
    Optimisé pour les volumes massifs (100k+ labels).
    """
    valid_labels = set()
    leaf_folders = []
    
    print(f"🔍 Analyse structurelle du dataset : {dataset_path}")

    # --- 1. COLLECTE DES LABELS (Folders + TXT) ---
    for root, dirs, files in os.walk(dataset_path):
        if files: 
            folder_name = os.path.basename(root).lower()
            leaf_folders.append(folder_name)
            valid_labels.add(folder_name)
            
        for f in files:
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as txt:
                        lines = [l.strip().lower() for l in txt.readlines() 
                                 if config.LABEL_MIN_LENGTH <= len(l.strip()) <= config.LABEL_MAX_LENGTH]
                        valid_labels.update(lines)
                except Exception:
                    pass

    # --- 2. FILTRAGE STATISTIQUE (Couche 3) ---
    blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset"]
    if leaf_folders and config.ENABLE_STATISTICAL_FALLBACK:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            if name.lower() in blacklist or (count/total > 0.15):
                if name.lower() in valid_labels:
                    valid_labels.remove(name.lower())

    # --- 3. VECTORISATION MASSIVE PAR BATCHS ---
    labels_to_embed = sorted(list(valid_labels))
    total_labels = len(labels_to_embed)
    
    if total_labels == 0:
        return {}

    batch_size = getattr(config, 'LABEL_BATCH_SIZE', 1000)
    print(f"🚀 Vectorisation de {total_labels} labels uniques (Batch Size: {batch_size})...")
    
    all_vectors = []
    for i in tqdm(range(0, total_labels, batch_size), desc="CLIP Label Encoding"):
        batch = labels_to_embed[i : i + batch_size]
        batch_vectors = embed_text_batch(batch)
        all_vectors.extend(batch_vectors)

    label_mapping = dict(zip(labels_to_embed, all_vectors))
    print(f"✅ Apprentissage terminé. {len(label_mapping)} concepts enregistrés.")
    return label_mapping