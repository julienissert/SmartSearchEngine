# src/intelligence/label_detector.py
import os
import re
from collections import Counter
import gc
from typing import Any
from src import config
from src.embeddings.text_embeddings import embed_text_batch 
from src.intelligence.handlers.raw_handler import resolve_raw_label
from src.intelligence.handlers.structured_handler import resolve_structured_label, is_label_noisy
from src.intelligence.llm_manager import  get_llm
from src.utils.logger import setup_logger
import pandas as pd
from src.ingestion.dispatcher import VISUAL_EXTENSIONS
logger = setup_logger("LabelDetector")

# Cache pour les correspondances explicites (filename -> label) [Niveau 0]
_FILE_MAPPING_CACHE = {}

def clear_memory():
    """Libère la RAM instantanément entre deux datasets."""
    global _FILE_MAPPING_CACHE
    _FILE_MAPPING_CACHE.clear()
    gc.collect()
    logger.info("RAM nettoyée : Prêt pour le prochain lot de données.")

def probe_path_strategy(df_sample, path_key, image_map):
    """Vérifie si les valeurs de la colonne existent dans l'index RAM."""
    if not path_key or df_sample is None or df_sample.empty or not image_map: 
        return None

    sample_rows = df_sample[path_key].dropna().head(5).astype(str).tolist()
    for test_val in sample_rows:
        target_name = test_val.strip().lower()
        if target_name in image_map:
            logger.info(f" IMAGE VALIDÉE : '{target_name}' trouvé via l'index.")
            return {"type": "map_index"}
    return None

def _discover_file_plan(path: str, ext: str, image_map: dict) -> dict | None:
    """
    Analyse de structure (Solo Test) : valide la qualité des colonnes.
    Si une colonne est bruyante (Code BNF), on appelle le LLM UNE SEULE FOIS.
    """
    filename = os.path.basename(path)
    path_col = None
    label_col = None

    try:
        sep = '\t' if ext.endswith('.tsv') else (',' if ext.endswith('.csv') else None)
        df_sample = pd.read_csv(path, nrows=10, sep=sep, engine='python', on_bad_lines='skip')
        if df_sample is None or df_sample.empty: return None

        # 1. ÉTAPE A : RECHERCHE DU CHEMIN (Preuve physique via Map RAM)
        for col in df_sample.columns:
            if probe_path_strategy(df_sample, col, image_map):
                path_col = col
                break 

        # 2. ÉTAPE B : RECHERCHE DU LABEL (Avec validation de bruit)
        magic_words = ["name", "label", "category", "title", "presentation", "chemical", "product"]
        for col in df_sample.columns:
            if col == path_col: continue 
            
            test_val = str(df_sample[col].iloc[0]).strip()
            if any(word in str(col).lower() for word in magic_words):
                if not is_label_noisy(test_val):
                    label_col = col
                    break
                else:
                    logger.warning(f" Colonne candidate '{col}' rejetée car contenu technique (Bruit).")
        
        # 3. ÉTAPE C : ARBITRAGE LLM UNIQUE (Seulement si doute ou bruit)
        if not label_col and get_llm().is_healthy():
            logger.info(f" Arbitrage LLM requis pour sceller le plan de : {filename}")
            res = get_llm().identify_csv_mapping(df_sample.to_string())
            if res and res.get("label_column"):
                label_col = res["label_column"]
                logger.info(f" Plan scellé par LLM -> Label: '{label_col}'")

        if label_col or path_col:
            return {
                "type": "column",
                "label_key": label_col or "unknown",
                "path_key": path_col, 
                "image_found": True if path_col else False
            }

    except Exception as e:
        logger.error(f"Erreur Discovery {filename}: {e}")
    return None

def analyze_dataset_structure(dataset_path):
    """
    Analyse globale Élite : crée une carte RAM des images et définit les plans 
    de mapping sans saturation disque.
    """
    clear_memory()
    global _FILE_MAPPING_CACHE
    valid_labels = set()
    leaf_folders = []
    file_plans = {}
    image_map = {}

    logger.info(f"Analyse sémantique du dataset : {dataset_path}...")

    # --- ÉTAPE 1 : SCAN UNIQUE & CONSTRUCTION DE L'INDEX ---
    for root, _, files in os.walk(dataset_path):
        if files: 
            leaf_folders.append(os.path.basename(root))
            
        for f in files:
            f_low = f.lower()
            path = os.path.join(root, f)
            
            if any(f_low.endswith(ve) for ve in VISUAL_EXTENSIONS):
                image_map[f_low] = os.path.abspath(path)

            if any(f_low.endswith(x) for x in [".txt", ".csv", ".tsv"]) and \
               any(x in f_low for x in ["label", "mapping", "meta"]):
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

    # --- ÉTAPE 2 : DISCOVERY DES PLANS (Utilise l'index RAM) ---
    for root, _, files in os.walk(dataset_path):
        for f in files:
            ext = f.lower()
            if any(ext.endswith(x) for x in [".txt", ".csv", ".tsv", ".json"]):
                path = os.path.join(root, f)
                plan = _discover_file_plan(path, ext, image_map)
                
                if plan:
                    abs_key = os.path.abspath(path).lower()
                    file_plans[abs_key] = plan

    # --- ÉTAPE 3 : FILTRAGE INTELLIGENT DES DOSSIERS (Niveau 1) ---
    if leaf_folders:
        counts = Counter(leaf_folders)
        for name, _ in counts.items():
            name_low = name.lower()
            if name_low not in config.TECHNICAL_FOLDERS and not is_label_noisy(name_low):
                valid_labels.add(name_low)

    # --- ÉTAPE 4 : GÉNÉRATION DES VECTEURS RÉFÉRENTS ---
    labels_list = [lbl for lbl in valid_labels if lbl]
    vectors_dict = {}
    if labels_list:
        logger.info(f"Génération des vecteurs pour {len(labels_list)} labels de référence...")
        vectors = embed_text_batch(labels_list)
        vectors_dict = dict(zip(labels_list, vectors))

    return {
        "vectors": vectors_dict,
        "file_plans": file_plans,
        "image_map": image_map 
    }
    # src/intelligence/label_detector.py

def detect_label(filepath, content, image=None, image_vector=None, label_mapping=None, suggested_label=None, type=None):
    """
    Dispatcher Universel : 
    Orchestre la détection du label par niveau de confiance mathématique.
    """
    fname = os.path.basename(filepath).lower()

    # --- NIVEAU 0 : Cache de Session (O(1)) ---
    # Si le fichier a déjà été identifié dans un mapping global (CSV/JSON de métadonnées)
    if fname in _FILE_MAPPING_CACHE:
        return _FILE_MAPPING_CACHE[fname]

    # --- NIVEAU 1 : DONNÉES STRUCTURÉES (Ligne CSV/JSON) ---
    if isinstance(content, dict):
        # On délègue au handler structuré qui suit le plan scellé par le LLM
        return resolve_structured_label(content, filepath, label_mapping, suggested_label)

    # --- NIVEAU 2 : DONNÉES RAW (Images, PDF, Fichiers Texte) ---
    # On extrait uniquement les vecteurs si le mapping est le dictionnaire complet du dataset
    vectors_only = label_mapping.get('vectors', {}) if isinstance(label_mapping, dict) else label_mapping
    
    # C'est ici que l'ARBITRAGE "Dossier vs Fichier" se produit
    label = resolve_raw_label(
        filepath=filepath, 
        text=content, 
        image=image, 
        image_vector=image_vector,  
        label_mapping=vectors_only, 
        suggested_label=suggested_label
    )

    # --- NIVEAU 3 : AFFINEMENT LLM (Dernier Recours) ---
    # Uniquement si le label est encore bruité ou inconnu après l'arbitrage physique
    if (is_label_noisy(label) or label == "unknown") and get_llm().is_healthy():
        # On ne tente l'affinement que si on a de la matière (OCR ou Image)
        if content or image:
            res = get_llm().refine_image_label(ocr_text=str(content), current_label=label)
            if res and res.get("refined_label"):
                return res["refined_label"]

    return label