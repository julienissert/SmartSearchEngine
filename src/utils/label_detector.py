# src/utils/label_detector.py
import os
import config
from collections import Counter
from embeddings.text_embeddings import embed_text

from .handlers.structured_handler import resolve_structured_label
from .handlers.raw_handler import detect_label

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
                        valid_labels.add(os.path.basename(root).lower()) # Label par dossier
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