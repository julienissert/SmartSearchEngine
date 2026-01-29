# src/intelligence/label_detector.py
import os
import re
from collections import Counter
import gc
from src import config
from src.embeddings.text_embeddings import embed_text_batch 
from src.intelligence.handlers.raw_handler import resolve_raw_label
from src.intelligence.handlers.structured_handler import resolve_structured_label
from src.intelligence.llm_manager import llm
from src.utils.logger import setup_logger

logger = setup_logger("LabelDetector")

# Cache pour les correspondances explicites (filename -> label) [Niveau 0]
_FILE_MAPPING_CACHE = {}

def clear_memory():
    """Libère la RAM instantanément entre deux datasets."""
    global _FILE_MAPPING_CACHE
    _FILE_MAPPING_CACHE.clear()
    gc.collect()
    logger.info("RAM nettoyée : Prêt pour le prochain lot de données.")

def is_label_noisy(label: str) -> bool:
    """Détecte si un label est un identifiant technique ou du bruit (Niveau 3)."""
    if not label or len(label) < 2: return True
    if label.isdigit() or str(label).startswith("auto_"): return True 
    if label.lower() in ["image", "img", "photo", "doc", "unknown", "document", "file"]: return True
    return False

def analyze_dataset_structure(dataset_path):
    """
    Analyse Élite : Scanne les dossiers et les fichiers de mapping (txt/csv).
    Résout le cas '1112.jpeg = concombre'.
    """
    clear_memory()
    global _FILE_MAPPING_CACHE
    valid_labels = set()
    leaf_folders = []
    
    logger.info(f"Analyse sémantique du dataset : {dataset_path}...")

    # 1. Collecte des labels et des Mappings Externes (Niveau 0)
    for root, _, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
        for f in files:
            if f.endswith((".txt", ".csv")) and any(x in f.lower() for x in ["label", "mapping", "meta"]):
                try:
                    with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                        for line in file:
                            if "=" in line or ":" in line:
                                parts = re.split(r'[=:]', line, maxsplit=1)
                                if len(parts) == 2:
                                    img_file = parts[0].strip().lower()
                                    label_val = parts[1].strip().lower()
                                    _FILE_MAPPING_CACHE[img_file] = label_val
                                    valid_labels.add(label_val)
                            # Fallback : une liste de labels par ligne
                            elif len(line.strip()) >= config.LABEL_MIN_LENGTH:
                                valid_labels.add(line.strip().lower())
                except Exception as e:
                    logger.warning(f"Erreur lecture mapping {f}: {e}")

    # 2. Filtrage intelligent des dossiers (Niveau 1)
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            name_low = name.lower()
            if name_low not in config.TECHNICAL_FOLDERS and (count/total < 0.15) and not is_label_noisy(name_low):
                valid_labels.add(name_low)

    # 3. Génération des vecteurs de référence
    labels_list = [lbl for lbl in valid_labels if lbl]
    if not labels_list: return {}

    logger.info(f"Génération des vecteurs pour {len(labels_list)} labels de référence...")
    vectors = embed_text_batch(labels_list)
    return dict(zip(labels_list, vectors))

def detect_label(filepath, content, image=None, image_vector=None, label_mapping=None, suggested_label=None, type=None):
    """
    Dispatcher Universel : Hiérarchie Niveau 0 à 3 avec arbitrage LLM.
    """
    fname = os.path.basename(filepath).lower()

    # --- NIVEAU 0 : Mémoire et Mapping Direct ---
    if fname in _FILE_MAPPING_CACHE:
        return _FILE_MAPPING_CACHE[fname]

    # --- NIVEAU 1 & 2 : Handlers (Structure & IA Rapide) ---
    if isinstance(content, dict):
        label = resolve_structured_label(content, filepath, label_mapping, suggested_label)
    else:
        label = resolve_raw_label(
            filepath=filepath, 
            text=content, 
            image=image, 
            image_vector=image_vector, 
            label_mapping=label_mapping, 
            suggested_label=suggested_label
    )

    # --- NIVEAU 3 : Arbitrage Consultant LLM ---
    if (is_label_noisy(label) or label == "unknown") and llm.is_healthy():
        if content or image:
            logger.info(f"Arbitrage LLM requis pour '{label}' ({os.path.basename(filepath)})")
            res = llm.refine_image_label(ocr_text=str(content), current_label=label)
            if res and res.get("refined_label"):
                refined = res["refined_label"]
                logger.info(f"Consultant LLM : Nouveau label validé -> {refined}")
                return refined

    return label