import os
import re
import numpy as np
from collections import Counter
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

def analyze_dataset_structure(dataset_path):
    valid_labels = set()
    leaf_folders = []
    
    print(f"Analyse du dataset {dataset_path}...")

    for root, dirs, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
        
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as txt:
                        lines = [l.strip() for l in txt.readlines() if l.strip()]
                        
                        if len(lines) > 2:
                            sample = lines[:20]
                            avg_len = sum(len(l.split()) for l in sample) / len(sample)
                            
                            if avg_len < 10:
                                print(f" -> Liste détectée : {f} ({len(lines)} entrées)")
                                for l in lines:
                                    if len(l) > 2:
                                        valid_labels.add(l.lower())
                except:
                    pass

    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        
        for name, count in counts.items():
            freq = count / total
            
            blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "archives", "dataset", "datasets"]
            
            is_structure = (freq > 0.1 and count > 5) or name.lower() in blacklist
            
            if not is_structure:
                if len(name) > 2 and not name.isdigit():
                    valid_labels.add(name.lower())

    print(f"Total labels appris : {len(valid_labels)}")
    return list(valid_labels)


def label_from_folder(filepath, valid_labels=None):
    dirname = os.path.dirname(filepath)
    folder = os.path.basename(dirname).lower()
    
    if not valid_labels:
        return folder if folder not in [".", ".."] else None

    if folder in valid_labels:
        return folder
    
    parent = os.path.basename(os.path.dirname(dirname)).lower()
    if parent in valid_labels:
        return parent
            
    return None

def label_from_text(text, known_labels=None):
    if not text: return None
    text_lower = text.lower()
    
    if known_labels:
        for lbl in sorted(known_labels, key=len, reverse=True):
            if lbl in text_lower:
                return lbl
    return None

def label_from_embeddings(image, candidate_labels):
    if not candidate_labels: return None
    try:
        img_emb = embed_image(image)
        if len(candidate_labels) > 500:
            return None 
        label_embs = [embed_text(lbl) for lbl in candidate_labels]
        sims = [np.dot(img_emb, emb) for emb in label_embs]
        return candidate_labels[int(np.argmax(sims))]
    except:
        return None

def auto_label_text(text):
    if not text: return "unknown_text"
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {"the", "a", "an", "of", "and", "in", "for", "le", "la", "les", "des", "du"}
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if not words: return "unknown_text"
    return f"auto_{Counter(words).most_common(1)[0][0]}"

def detect_label(filepath=None, text=None, image=None, known_labels=None):
    label = None
    if filepath: label = label_from_folder(filepath, valid_labels=known_labels)
    if not label and text: label = label_from_text(text, known_labels)
    if not label and image and known_labels: label = label_from_embeddings(image, known_labels)
    if not label:
        if text: label = auto_label_text(text)
        elif image: label = "auto_img"
        else: label = "unknown"
    return label