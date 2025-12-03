import os
import re
import numpy as np
from collections import Counter
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

# --- ANALYSE GLOBALE ---

def analyze_dataset_structure(dataset_path):
    valid_labels = set()
    leaf_folders = []
    
    print(f"Analyse du dataset {dataset_path}...")

    for root, dirs, files in os.walk(dataset_path):
        # 1. Collecte des dossiers feuilles
        if files: 
            leaf_folders.append(os.path.basename(root))
        
        # 2. Détection de listes 
        for f in files:
            if f.endswith(".txt"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as txt:
                        lines = [l.strip() for l in txt.readlines() if l.strip()]
                        
                        if len(lines) > 50:
                            avg_len = sum(len(l.split()) for l in lines[:20]) / 20
                            if avg_len < 10:
                                print(f" -> Liste détectée : {f} ({len(lines)} entrées)")
                                for l in lines:
                                    if len(l) > 2:
                                        valid_labels.add(l.lower())
                except:
                    pass

    # 3. Filtrage statistique des dossiers
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        
        for name, count in counts.items():
            freq = count / total
            # On ignore si > 10% du dataset (structure) ou mot technique
            is_structure = (freq > 0.1 and count > 5) or name.lower() in ["images", "img", "photos", "train", "test", "meta"]
            
            if not is_structure:
                if len(name) > 2 and not name.isdigit():
                    valid_labels.add(name.lower())

    print(f"Total labels appris : {len(valid_labels)}")
    return list(valid_labels)

# --- DÉTECTION UNITAIRE ---

def label_from_folder(filepath, valid_labels=None):
    dirname = os.path.dirname(filepath)
    folder = os.path.basename(dirname).lower()
    
    if not valid_labels:
        return folder if folder not in [".", ".."] else None

    if folder in valid_labels:
        return folder
    
    # Test dossier parent si le courant est ignoré
    parent = os.path.basename(os.path.dirname(dirname)).lower()
    if parent in valid_labels:
        return parent
            
    return None

def label_from_text(text, known_labels=None):
    if not text: return None
    text_lower = text.lower()
    
    if known_labels:
        # Trie par longueur pour matcher les termes précis d'abord
        for lbl in sorted(known_labels, key=len, reverse=True):
            if lbl in text_lower:
                return lbl
    return None

def label_from_embeddings(image, candidate_labels):
    if not candidate_labels: return None
    try:
        img_emb = embed_image(image)
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
    
    # 1. Dossier
    if filepath:
        label = label_from_folder(filepath, valid_labels=known_labels)
    
    # 2. Texte OCR / Contenu
    if not label and text:
        label = label_from_text(text, known_labels)
    
    # 3. Vision (CLIP) 
    if not label and image and known_labels and len(known_labels) < 500:
        label = label_from_embeddings(image, known_labels)
    
    # 4. Fallback
    if not label:
        if text: label = auto_label_text(text)
        elif image: label = "auto_img"
        else: label = "unknown"

    return label