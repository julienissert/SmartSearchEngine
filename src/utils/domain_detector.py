# src/utils/domain_detector.py
import numpy as np
import os
from PIL import Image
import config  
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

# 1. TEMPLATES pour l'Audit et le Fallback (Analyse sémantique complexe)
PROMPT_TEMPLATES = [
    "A photo of {}.", "Category: {}.", "This is related to {}.",
    "A list of {}.", "Technical data about {}."
]

DOMAIN_VECTORS_AUDIT = {}
DOMAIN_VECTORS_RAW = {}

for domain in config.TARGET_DOMAINS:
    # Vecteurs pour l'IA (Audit) : avec phrases pour mieux comprendre les images
    prompts = [template.format(domain) for template in PROMPT_TEMPLATES]
    embeddings = [embed_text(p) for p in prompts]
    vec_audit = np.mean(embeddings, axis=0)
    DOMAIN_VECTORS_AUDIT[domain] = vec_audit / np.linalg.norm(vec_audit)
    
    # Vecteurs pour le Dossier (Décision) : mot pur pour un match à 100%
    vec_raw = embed_text(domain)
    DOMAIN_VECTORS_RAW[domain] = vec_raw / np.linalg.norm(vec_raw)

def get_raw_similarity(word):
    """Calcule la similarité brute entre un mot et les noms de domaines."""
    if not word or len(str(word).strip()) < 2: return None
    emb = embed_text(str(word))
    norm = np.linalg.norm(emb)
    if norm == 0: return None
    emb = emb / norm
    return {d: float(np.dot(emb, v)) for d, v in DOMAIN_VECTORS_RAW.items()}

def detect_domain(text: str = None, pil_image: Image.Image = None, filepath: str = None, content_dict: dict = None):
    """
    Hiérarchie Pro : 
    - Dossier (C0) & Structure (C1) utilisent des vecteurs bruts (Décision déterministe).
    - L'Audit (Étape A) utilise des vecteurs templates (Analyse de contenu).
    """
    
    # --- ÉTAPE A : CALCUL SYSTÉMATIQUE DE L'AUDIT IA (Contenu brut) ---
    ai_probabilities = {d: 0.0 for d in config.TARGET_DOMAINS}
    doc_emb = embed_image(pil_image) if pil_image else embed_text(text) if text else None
    
    if doc_emb is not None:
        doc_norm = np.linalg.norm(doc_emb)
        if doc_norm > 0:
            doc_emb = doc_emb / doc_norm
            # On utilise l'IA (Audit) avec templates ici
            raw_scores = {d: float(np.dot(doc_emb, v)) for d, v in DOMAIN_VECTORS_AUDIT.items()}
            logits = np.array(list(raw_scores.values())) * 100.0
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            ai_probabilities = {k: float(p) for k, p in zip(raw_scores.keys(), probs)}

    # --- ÉTAPE B : HIÉRARCHIE DES COUCHES DE DÉCISION ---

    # COUCHE 0 : Contexte de Dossier par Segmentation
    if filepath:
        try:
            rel_path = os.path.relpath(filepath, config.DATASET_DIR)
            path_context = os.path.splitext(rel_path)[0].replace(os.sep, " ").replace("-", " ").replace("_", " ")
            segments = path_context.split()
            
            for segment in segments:
                if len(segment) < 2: continue
                
                # Comparaison directe Mot vs Domaine (sans templates pollueurs)
                scores = get_raw_similarity(segment)
                if scores:
                    max_val = max(scores.values())
                    if max_val > 0.85: 
                        best_path = max(scores, key=lambda k: scores[k])
                        return best_path, ai_probabilities, "path_forced"
        except Exception:
            pass

    # COUCHE 1 : Signature de Structure (Clés/Colonnes)
    if isinstance(content_dict, dict):
        schema_text = " ".join(content_dict.keys())
        # On teste les mots de la structure
        for word in schema_text.replace("_", " ").split():
            scores = get_raw_similarity(word)
            if scores and max(scores.values()) > 0.85:
                best_schema = max(scores, key=lambda k: scores[k])
                return best_schema, ai_probabilities, "schema_forced"

    # COUCHE 2 : Fallback IA (Si aucun indice dans le dossier/structure)
    best_ai = max(ai_probabilities, key=lambda k: ai_probabilities[k])
    return best_ai, ai_probabilities, "ai_content"