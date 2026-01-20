# src/utils/label_detector.py
import os
import config
from collections import Counter
from embeddings.text_embeddings import embed_text_batch # Utilisation du batching obligatoire ici
from .handlers.structured_handler import resolve_structured_label
from .handlers.raw_handler import detect_label

def analyze_dataset_structure(dataset_path):
    """Apprend les labels et les vectorise par lots pour une performance maximale."""
    valid_labels = set()
    leaf_folders = []
    
    print(f"Analyse structurelle : {dataset_path}...")

    # 1. Collecte des labels (Dossiers + fichiers TXT)
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
                except: pass

    # Filtrage statistique des dossiers
    blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset"]
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            if name.lower() not in blacklist and (count/total < 0.15):
                valid_labels.add(name.lower())

    # 2. Vectorisation massive (Bypass de la boucle séquentielle)
    labels_to_embed = [lbl for lbl in valid_labels if lbl]
    total_labels = len(labels_to_embed)
    
    if total_labels == 0:
        return {}

    print(f" Vectorisation CLIP de {total_labels} labels (Mode Batch)...")
    
    # embed_text_batch gère déjà le passage au GPU/CPU de manière groupée
    vectors = embed_text_batch(labels_to_embed)
    
    # Reconstruction du dictionnaire de mapping
    label_mapping = dict(zip(labels_to_embed, vectors))
    
    print(f" Apprentissage terminé. {total_labels} concepts mémorisés.")
    return label_mapping