# src/intelligence/handlers/structured_handler.py
import os
import numpy as np
from src import config
from src.embeddings.text_embeddings import embed_text
from src.intelligence.llm_manager import llm
from src.utils.logger import setup_logger
import json

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


def is_valid_label_format(value) -> bool:
    """Vérifie si la valeur a la forme d'un label (pas un prix, pas trop long)."""
    val_str = str(value).strip()
    if not val_str or val_str.lower() == "nan": return False
    # Exclure les prix/ID (numérique pur)
    if val_str.replace('.', '', 1).isdigit(): return False
    # Exclure les descriptions infinies
    if not (config.LABEL_MIN_LENGTH <= len(val_str) <= 100): return False
    return True

def resolve_structured_label(data_dict, source_path, label_mapping=None, suggested_label=None, dataset_name=None):
    """
    Stratégie Hybride Élite : Statistique + Sémantique + Consultant LLM.
    """
    load_cache()
    
    # --- COUCHE 0 : Validation Immédiate ---
    if suggested_label:
        return str(suggested_label).lower().strip()

    # Identifiant unique du dataset (Dossier parent + Nom interne pour H5)
    dataset_id = os.path.dirname(source_path)
    cache_key = f"{dataset_id}::{dataset_name}" if dataset_name else dataset_id
    
    # --- NIVEAU 0 : Utilisation du Cache (Vitesse maximale) ---
    if cache_key in _SCHEMA_CACHE:
        target_col = _SCHEMA_CACHE[cache_key]
        if target_col and data_dict is not None and target_col in data_dict:
            return str(data_dict[target_col]).lower().strip()

    # Sécurité si le loader a renvoyé du vide
    if data_dict is None:
        return "unknown"
    
    # --- NIVEAU 1 : Intelligence Rapide (Heuristiques & Vocabulaire) ---
    keys = list(data_dict.keys())
    
    # Test 1 : Correspondance exacte avec ton dictionnaire de vérité
    if label_mapping is not None: 
        for key, value in data_dict.items():
            val_clean = str(value).lower().strip()
            if val_clean in label_mapping: 
                _SCHEMA_CACHE[cache_key] = key
                return val_clean

    # Test 2 : Mots-clés "Magiques" dans les en-têtes
    magic_words = ["name", "label", "category", "titre", "product", "nom", "item"]
    for key in keys:
        if any(word in str(key).lower() for word in magic_words):
            if is_valid_label_format(data_dict[key]):
                _SCHEMA_CACHE[cache_key] = key
                return str(data_dict[key]).lower().strip()

    # --- NIVEAU 2 : Consultant LLM (Dernier recours / Discovery) ---
    # Si les méthodes rapides échouent, on demande au LLM d'analyser le schéma
    if llm.is_healthy():
        logger.info(f"Analyse de schéma requise pour le dataset : {cache_key}")
        sample = str(list(data_dict.items()))
        res = llm.identify_csv_mapping(sample)
        
        if res and res.get("label_column"):
            target_col = res["label_column"]
            _SCHEMA_CACHE[cache_key] = target_col
            save_cache()
            logger.info(f"Consultant LLM : Colonne identifiée -> '{target_col}'")
            return str(data_dict.get(target_col, "unknown")).lower().strip()

    # --- NIVEAU 3 : Profilage Statistique (Filet de sécurité final) ---
    for key, value in data_dict.items():
        if is_valid_label_format(value):
            _SCHEMA_CACHE[cache_key] = key
            return str(value).lower().strip()

    return "unknown"