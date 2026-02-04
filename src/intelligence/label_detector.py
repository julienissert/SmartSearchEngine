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
import pandas as pd
logger = setup_logger("LabelDetector")

# Cache pour les correspondances explicites (filename -> label) [Niveau 0]
_FILE_MAPPING_CACHE = {}

def clear_memory():
    """Libère la RAM instantanément entre deux datasets."""
    global _FILE_MAPPING_CACHE
    _FILE_MAPPING_CACHE.clear()
    gc.collect()
    logger.info("RAM nettoyée : Prêt pour le prochain lot de données.")

def is_label_noisy(label) -> bool:
    """
    Détecte si un label est un identifiant technique ou du bruit (Niveau 3).
    Gère les types numpy.int64 envoyés par pandas.
    """
    val = str(label).strip()
    
    if not val or len(val) < 2: return True
    
    if val.isdigit(): return True 

    if val.lower() in ["image", "img", "photo", "doc", "unknown", "document", "file"]: 
        return True       
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
    file_plans = {}
    
    logger.info(f"Analyse sémantique du dataset : {dataset_path}...")

    # 1. Collecte des labels et des Mappings Externes (Niveau 0)
    for root, _, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
            
        for f in files:
            path = os.path.join(root, f)
            ext = f.lower()
            
            # --- 1. CAS A : FICHIERS DE MAPPING TECHNIQUE (img=label) ---
            if any(ext.endswith(x) for x in [".txt", ".csv", ".tsv"]) and any(x in f.lower() for x in ["label", "mapping", "meta"]):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as file:
                        for line in file:
                            if "=" in line or ":" in line:
                                parts = re.split(r'[=:]', line, maxsplit=1)
                                if len(parts) == 2:
                                    img_file = parts[0].strip().lower()
                                    label_val = parts[1].strip().lower()
                                    _FILE_MAPPING_CACHE[img_file] = label_val
                                    valid_labels.add(label_val)
                            elif len(line.strip()) >= config.LABEL_MIN_LENGTH:
                                valid_labels.add(line.strip().lower())
                except Exception as e:
                    logger.warning(f"Erreur lecture mapping {f}: {e}")
                continue 

            # --- 2. CAS B : DISCOVERY POUR DONNÉES STRUCTURÉES ---
            if any(ext.endswith(x) for x in [".txt", ".csv", ".tsv"]):
                abs_key = os.path.abspath(path).lower()
                plan = None

                try:
                    # Étape 1 : Heuristique rapide (CSV/TSV uniquement)
                    if ".txt" not in ext:
                        sep = '\t' if ext.endswith('.tsv') else (',' if ext.endswith('.csv') else None)
                        
                        df_sample = pd.read_csv(path, nrows=5, sep=sep, engine='python')
                        
                        magic_words = ["name", "label", "category", "titre", "product", "nom", "item"]
                        for col in df_sample.columns:
                            if any(word in str(col).lower() for word in magic_words):
                                test_val = df_sample[col].iloc[0] if not df_sample[col].empty else ""
                                
                                if not is_label_noisy(test_val):
                                    plan = {"type": "column", "key": col}
                                    logger.info(f" Heuristique validée : Colonne '{col}' pour {f}")
                                    break
                                else:
                                    logger.warning(f" Heuristique rejetée : '{col}' contient du bruit ('{test_val}')")
                    
                    # Étape 2 : LLM Discovery (Fallback si heuristique rejetée ou si c'est un TXT)
                    if not plan and llm.is_healthy():
                        with open(path, 'r', encoding='utf-8', errors='ignore') as tmp:
                            sample = "".join([tmp.readline() for _ in range(10)])
                        
                        logger.info(f" Discovery LLM (Solo Test) requis pour : {f}")
                        ext_name = ".txt" if ext.endswith(".txt") else (".csv" if ext.endswith(".csv") else ".tsv")
                        plan = llm.identify_mapping_plan(sample, ext_name)

                    if plan:
                        file_plans[abs_key] = plan
                        logger.info(f" Plan LLM validé pour {f} -> Colonne fixée : '{plan.get('key')}'")
                except Exception as e:
                    logger.error(f"Erreur durant la Discovery de {f}: {e}")
                    continue

    # 3. Filtrage intelligent des dossiers (Niveau 1)
    if leaf_folders:
        counts = Counter(leaf_folders)
        total = len(leaf_folders)
        for name, count in counts.items():
            name_low = name.lower()
            if name_low not in config.TECHNICAL_FOLDERS and (count/total < 0.15) and not is_label_noisy(name_low):
                valid_labels.add(name_low)

    # 4. Génération des vecteurs et renvoi du pack complet
    labels_list = [lbl for lbl in valid_labels if lbl]
    vectors_dict = {}
    if labels_list:
        logger.info(f"Génération des vecteurs pour {len(labels_list)} labels de référence...")
        vectors = embed_text_batch(labels_list)
        vectors_dict = dict(zip(labels_list, vectors))

    return {
        "vectors": vectors_dict,
        "file_plans": file_plans
    }

def detect_label(filepath, content, image=None, image_vector=None, label_mapping=None, suggested_label=None, type=None):
    """
    Dispatcher Universel : Hiérarchie Niveau 0 à 3 avec arbitrage LLM.
    """
    fname = os.path.basename(filepath).lower()

    # --- NIVEAU 0 : Mémoire et Mapping Direct ---
    if fname in _FILE_MAPPING_CACHE:
        return _FILE_MAPPING_CACHE[fname]

    vectors_only = label_mapping.get('vectors', {}) if isinstance(label_mapping, dict) and 'vectors' in label_mapping else label_mapping
    # --- NIVEAU 1 & 2 : Handlers (Structure & IA Rapide) ---
    if isinstance(content, dict):
        label = resolve_structured_label(content, filepath, label_mapping, suggested_label)
    else:
        label = resolve_raw_label(
            filepath=filepath, 
            text=content, 
            image=image, 
            image_vector=image_vector, 
            label_mapping=vectors_only, 
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