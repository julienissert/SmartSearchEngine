# src/utils/label_detector.py
import os
from collections import Counter
from src import config
from src.embeddings.text_embeddings import embed_text_batch 
from .handlers.raw_handler import resolve_raw_label
from .handlers.structured_handler import resolve_structured_label

def analyze_dataset_structure(dataset_path):
    """Analyse la structure pour créer un vocabulaire de labels de référence."""
    valid_labels = set()
    leaf_folders = []
    
    print(f"Analyse sémantique du dataset : {dataset_path}...")

    # 1. Collecte des labels potentiels 
    for root, dirs, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
        for f in files:
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as txt:
                        lines = [l.strip().lower() for l in txt.readlines() 
                                 if len(l.strip()) >= config.LABEL_MIN_LENGTH]
                        valid_labels.add(os.path.basename(root).lower()) 
                        valid_labels.update(lines)
                except Exception:
                    pass

    # 2. Filtrage intelligent 
    blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset"]
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            if name.lower() not in blacklist and (count/total < 0.15):
                valid_labels.add(name.lower())

    # 3. GÉNÉRATION MASSIVE DES VECTEURS 
    labels_list = [lbl for lbl in valid_labels if lbl]
    if not labels_list:
        return {}

    print(f" Génération des vecteurs pour {len(labels_list)} labels...")
    vectors = embed_text_batch(labels_list)
    return dict(zip(labels_list, vectors))

def detect_label(filepath, content, image=None, label_mapping=None, suggested_label=None, type=None):
    """
    Dispatcher Universel : Oriente vers la bonne stratégie de résolution.
    """
    if isinstance(content, dict):
        # Cas CSV / H5
        return resolve_structured_label(
            data_dict=content,
            source_path=filepath,
            label_mapping=label_mapping,
            suggested_label=suggested_label,
            dataset_name=content.get('dataset_name') if type == 'h5' else None
        )
    
    # Cas PDF / Image / TXT
    return resolve_raw_label(
        filepath=filepath,
        text=str(content) if content else None,
        image=image,
        label_mapping=label_mapping,
        suggested_label=suggested_label
    )