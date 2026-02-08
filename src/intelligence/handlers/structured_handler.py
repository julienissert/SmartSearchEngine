# src/intelligence/handlers/structured_handler.py
import os
import numpy as np
from src import config
from src.embeddings.text_embeddings import embed_text
from src.intelligence.llm_manager import get_llm
from src.utils.logger import setup_logger
import json
from src.config import DATASET_DIR

logger = setup_logger("StructuredHandler")
CACHE_FILE = config.SCHEMA_CACHE_PATH

# --- ÉTAT GLOBAL (Type Safe) ---
_SCHEMA_CACHE = {}
_CACHE_LOADED = False

def reset_memory():
    """Force la remise à zéro de la mémoire vive."""
    global _SCHEMA_CACHE, _CACHE_LOADED
    _SCHEMA_CACHE = {}
    _CACHE_LOADED = False
    
def load_cache():
    global _SCHEMA_CACHE, _CACHE_LOADED
    if _CACHE_LOADED: 
        return 
    
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _SCHEMA_CACHE.update(data)
        except Exception:
            _SCHEMA_CACHE = {}
    
    _CACHE_LOADED = True

def save_cache():
    """Sauvegarde immédiate pour survivre aux crashs."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_SCHEMA_CACHE, f, indent=4)
    except Exception as e:
        logger.error(f"Erreur sauvegarde mémoire schémas : {e}")


def is_label_noisy(label) -> bool:
    """
    Détecte si un label est un identifiant technique (codes BNF, IDs) 
    ou du bruit de manière restrictive.
    """
    val = str(label).strip()
    
    # 1. Vide ou trop court
    if not val or len(val) < 3: return True
    
    # 2. Chiffres purs ou prix (Années, Quantités)
    if val.replace('.', '', 1).isdigit(): return True 

    # 3. DÉTECTION DE CODE TECHNIQUE (ex: 0601023abaaaaaa)
    if " " not in val and any(c.isdigit() for c in val):
        return True 

    # 4. RATIO DE CHIFFRES (Un label humain a peu de chiffres)
    digit_count = sum(c.isdigit() for c in val)
    if digit_count / len(val) > 0.15: # Plus de 25% de chiffres = probable ID
        return True

    # 5. Mots interdits techniques
    if val.lower() in ["image", "img", "photo", "doc", "unknown", "document", "file", "nan", "null"]: 
        return True 
              
    return False

def resolve_structured_label(data_dict, source_path, label_mapping=None, suggested_label=None, dataset_name=None):
    load_cache()
    if suggested_label: return str(suggested_label).lower().strip()
    
    # 1. Priorité au Plan scellé (On utilise 'label_key')
    if isinstance(label_mapping, dict) and 'file_plans' in label_mapping:
        abs_path = os.path.abspath(source_path).lower()
        plan = label_mapping['file_plans'].get(abs_path)
        if plan and plan.get('label_key') in data_dict:
            return str(data_dict[plan['label_key']]).lower().strip()

    # 2. Cache de session
    cache_key = os.path.dirname(source_path)
    if cache_key in _SCHEMA_CACHE:
        col = _SCHEMA_CACHE[cache_key]
        if col in data_dict: return str(data_dict[col]).lower().strip()

    # 3. Mots Magiques + Validation STRICTE
    magic_words = ["presentation", "chemical", "name", "label", "category", "titre", "product", "nom"]
    for key in data_dict.keys():
        if any(word in str(key).lower() for word in magic_words):
            if is_label_noisy(data_dict[key]):
                _SCHEMA_CACHE[cache_key] = key
                save_cache()
                return str(data_dict[key]).lower().strip()

    # 4. Fallback LLM (avec nettoyage des guillemets)
    if get_llm().is_healthy():
        res = get_llm().identify_csv_mapping(str(list(data_dict.items())[:5]))
        if res and res.get("label_column"):
            target_col = res["label_column"].strip("'\" ")
            _SCHEMA_CACHE[cache_key] = target_col
            save_cache()
            return str(data_dict.get(target_col, "unknown")).lower().strip()

    return "unknown"

def resolve_image_path(record: dict, source_path: str, plan: dict) -> str | None:
    path_key = plan.get("path_key")
    dataset_root = plan.get("dataset_root")
    
    if not path_key or not dataset_root: return None
        
    img_val = str(record.get(path_key, "")).strip()
    if not img_val or img_val.lower() in ["nan", "none", ""]: return None
    
    target_name = img_val.lower()

    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.lower() == target_name:
                return os.path.join(root, f)
    return None