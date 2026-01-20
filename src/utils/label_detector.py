# src/utils/label_detector.py
import os
import config
from collections import Counter
from tqdm import tqdm
from embeddings.text_embeddings import embed_text_batch # Import de la version batchée

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
        # On enregistre les noms de dossiers qui contiennent des fichiers
        if files: 
            folder_name = os.path.basename(root).lower()
            leaf_folders.append(folder_name)
            valid_labels.add(folder_name)
            
        # Extraction des labels supplémentaires dans les fichiers .txt
        for f in files:
            if f.endswith(".txt"):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as txt:
                        # Filtrage par longueur pour éviter les bruits
                        lines = [l.strip().lower() for l in txt.readlines() 
                                 if config.LABEL_MIN_LENGTH <= len(l.strip()) <= config.LABEL_MAX_LENGTH]
                        valid_labels.update(lines)
                except Exception as e:
                    pass

    # --- 2. FILTRAGE STATISTIQUE (Couche 3) ---
    # Élimine les dossiers techniques (images, meta, etc.) s'ils sont trop fréquents
    blacklist = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset"]
    if leaf_folders and config.ENABLE_STATISTICAL_FALLBACK:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            # Si un nom de dossier représente plus de 15% du dataset, c'est probablement structurel et non un label
            if name.lower() in blacklist or (count/total > 0.15):
                if name.lower() in valid_labels:
                    valid_labels.remove(name.lower())

    # --- 3. VECTORISATION MASSIVE PAR BATCHS ---
    labels_to_embed = sorted(list(valid_labels)) # Tri pour la cohérence
    total_labels = len(labels_to_embed)
    
    if total_labels == 0:
        return {}

    # On utilise LABEL_BATCH_SIZE (ex: 1000) pour ne pas saturer la RAM/VRAM
    # Si non défini dans config.py, on utilise 1000 par défaut
    batch_size = getattr(config, 'LABEL_BATCH_SIZE', 1000)
    
    print(f"🚀 Vectorisation de {total_labels} labels uniques (Batch Size: {batch_size})...")
    
    all_vectors = []
    
    # Barre de progression pour le suivi des 100k labels
    for i in tqdm(range(0, total_labels, batch_size), desc="CLIP Label Encoding"):
        batch = labels_to_embed[i : i + batch_size]
        
        # Appel massif au modèle CLIP
        # Cette fonction doit retourner une liste de vecteurs (list of np.array)
        batch_vectors = embed_text_batch(batch)
        all_vectors.extend(batch_vectors)

    # Création du dictionnaire final : { "nom_label": vecteur_faiss }
    label_mapping = dict(zip(labels_to_embed, all_vectors))
    
    print(f"✅ Apprentissage terminé. {len(label_mapping)} concepts enregistrés en mémoire.")
    return label_mapping