# src/utils/domain_detector.py
import numpy as np
import os
from PIL import Image
from src import config  
from src.embeddings.text_embeddings import embed_text
from src.embeddings.image_embeddings import embed_image

# On garde tes templates pro
PROMPT_TEMPLATES = [
    "A photo of {}.", "Category: {}.", "This is related to {}.",
    "A list of {}.", "Technical data about {}."
]

DOMAIN_VECTORS_AUDIT = {}

def init_domain_references():
    """Initialise les centroïdes de domaine via les templates."""
    global DOMAIN_VECTORS_AUDIT
    if DOMAIN_VECTORS_AUDIT: return 
    
    for domain in config.TARGET_DOMAINS:
        # On crée une empreinte large (Centroïde) pour chaque domaine
        prompts = [template.format(domain) for template in PROMPT_TEMPLATES]
        embeddings = [embed_text(p) for p in prompts]
        vec_audit = np.mean(embeddings, axis=0)
        DOMAIN_VECTORS_AUDIT[domain] = vec_audit / np.linalg.norm(vec_audit)

def detect_domain(
    text: str | None = None, 
    pil_image: Image.Image | None = None, 
    filepath: str | None = None, 
    content_dict: dict | None = None, 
    precomputed_vector: np.ndarray | None = None
):
    """
    Détecteur de domaine pro à 3 couches.
    Optimisé pour capturer le signal sémantique des datasets complexes.
    """
    init_domain_references()
    
    # --- PRÉPARATION : Création du vecteur de contexte (Chemin + Structure) ---
    context_vec = None
    context_parts = []
    
    if filepath:
        # On extrait tout le texte utile du chemin
        path_text = os.path.basename(filepath).replace("_", " ").replace("-", " ")
        context_parts.append(path_text)
    
    if isinstance(content_dict, dict):
        # On extrait toutes les clés (le schéma du dataset)
        schema_text = " ".join(content_dict.keys()).replace("_", " ")
        context_parts.append(schema_text)

    if context_parts:
        # On crée un vecteur global qui représente le "sens" de la structure
        full_context = " ".join(context_parts)
        raw_context_vec = embed_text(full_context)
        context_vec = raw_context_vec / np.linalg.norm(raw_context_vec)

    # --- ÉTAPE A : CALCUL DES SCORES PAR COUCHE ---
    
    # 1. Score Structurel (Couche 0 & 1 combinées par signal cumulé)
    struct_scores = {d: 0.0 for d in config.TARGET_DOMAINS}
    if context_vec is not None:
        struct_scores = {d: float(np.dot(context_vec, v)) for d, v in DOMAIN_VECTORS_AUDIT.items()}

    # 2. Score IA / CLIP (Couche 2)
    ai_scores = {d: 0.0 for d in config.TARGET_DOMAINS}
    doc_emb = precomputed_vector
    if doc_emb is None:
        doc_emb = embed_image(pil_image) if pil_image else embed_text(text) if text else None
    
    if doc_emb is not None:
        norm_vec = doc_emb / np.linalg.norm(doc_emb)
        ai_scores = {d: float(np.dot(norm_vec, v)) for d, v in DOMAIN_VECTORS_AUDIT.items()}

    # --- ÉTAPE B : LOGIQUE DE DÉCISION HIÉRARCHIQUE ---

    # PRIORITÉ 1 : Si la structure (clés CSV/nom de fichier) est claire (> 0.50)
    best_struct = max(struct_scores, key=lambda k: struct_scores[k], default="unknown")
    
    # On vérifie si on a un domaine valide avant de tester le score
    if best_struct != "unknown" and struct_scores[best_struct] > 0.50:
        return best_struct, struct_scores, "structural_context_forced"

    # PRIORITÉ 2 : Fusion sémantique (AI + Structure)
    final_probs = {}
    for d in config.TARGET_DOMAINS:
        combined_score = (struct_scores[d] * 0.6) + (ai_scores[d] * 0.4)
        final_probs[d] = combined_score

    # Softmax léger pour l'exportation des probabilités (température=10.0)
    logits = np.array(list(final_probs.values())) * 10.0
    exp_logits = np.exp(logits - np.max(logits))
    probs_values = exp_logits / np.sum(exp_logits)
    ai_probabilities = {k: float(p) for k, p in zip(final_probs.keys(), probs_values)}
    
    best_final = max(final_probs, key=lambda k: final_probs[k], default="unknown")
    
    if best_final == "unknown" or final_probs[best_final] < 0.25:
        return "unknown", ai_probabilities, "low_confidence_rejection"

    return best_final, ai_probabilities, "semantic_fusion"