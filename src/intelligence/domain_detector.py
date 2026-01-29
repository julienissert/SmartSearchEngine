# src/intelligence/domain_detector.py
import numpy as np
import os
from PIL import Image
from src import config  
from src.embeddings.text_embeddings import embed_text
from src.embeddings.image_embeddings import embed_image
from src.intelligence.llm_manager import llm  
from src.utils.logger import setup_logger

logger = setup_logger("DomainDetector")

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
    Détecteur de domaine pro à 3 couches + Arbitrage LLM par Multi-Gap Analysis.
    Optimisé pour la haute échelle (15+ domaines).
    """
    init_domain_references()
    
    # --- 1. PRÉPARATION DU CONTEXTE ---
    context_vec = None
    context_parts = []
    
    if filepath:
        path_text = os.path.basename(filepath).replace("_", " ").replace("-", " ")
        context_parts.append(path_text)
    
    if isinstance(content_dict, dict):
        schema_text = " ".join(content_dict.keys()).replace("_", " ")
        context_parts.append(schema_text)

    if context_parts:
        raw_context_vec = embed_text(" ".join(context_parts))
        context_vec = raw_context_vec / np.linalg.norm(raw_context_vec)

    # --- 2. CALCUL DES SCORES (STRUCTURE + IA) ---
    struct_scores = {d: 0.0 for d in config.TARGET_DOMAINS}
    if context_vec is not None:
        struct_scores = {d: float(np.dot(context_vec, v)) for d, v in DOMAIN_VECTORS_AUDIT.items()}

    ai_scores = {d: 0.0 for d in config.TARGET_DOMAINS}
    doc_emb = precomputed_vector
    if doc_emb is None:
        doc_emb = embed_image(pil_image) if pil_image else embed_text(text) if text else None
    
    if doc_emb is not None:
        norm_vec = doc_emb / np.linalg.norm(doc_emb)
        ai_scores = {d: float(np.dot(norm_vec, v)) for d, v in DOMAIN_VECTORS_AUDIT.items()}

    # --- 3. LOGIQUE DE DÉCISION & PROBABILITÉS ---
    # Priorité structurelle forte (> 0.50)
    best_struct = max(struct_scores, key=lambda k: struct_scores[k], default="unknown")
    if best_struct != "unknown" and struct_scores[best_struct] > 0.50:
        return best_struct, struct_scores, "structural_context_forced"

    # Fusion sémantique (60% Structure / 40% Contenu)
    final_probs = {d: (struct_scores[d] * 0.6) + (ai_scores[d] * 0.4) for d in config.TARGET_DOMAINS}

    # Softmax avec facteur 10.0 pour polariser la distribution
    logits = np.array(list(final_probs.values())) * 10.0
    exp_logits = np.exp(logits - np.max(logits))
    probs_values = exp_logits / np.sum(exp_logits)
    ai_probabilities = {k: float(p) for k, p in zip(final_probs.keys(), probs_values)}
    
    best_final = max(ai_probabilities, key=lambda k: ai_probabilities[k], default="unknown")
    max_prob = ai_probabilities.get(best_final, 0.0)

    # --- 4. ANALYSE DE MARGE MULTIPLE (Le "Peloton de tête") ---
    # On identifie tous les candidats dont le score est proche du meilleur (marge < 0.15)
    # et qui représentent un signal réel (> 0.05).
    shortlist = {
        d: round(p, 4) for d, p in ai_probabilities.items() 
        if (max_prob - p) < 0.15 and p > 0.05
    }

    # --- 5. ARBITRAGE LLM (Conditions de Confusion) ---
    # On arbitre si confiance faible OU si plusieurs domaines sont en compétition serrée.
    low_confidence = (max_prob < 0.25)
    is_confused = (len(shortlist) > 1)

    if (low_confidence or is_confused) and (text or filepath) and llm.is_healthy():
        logger.info(
            f"Arbitrage requis | Confusion entre {len(shortlist)} domaines "
            f"(Top: {best_final} @ {max_prob:.2f})."
        )
        
        result = llm.arbitrate_domain(
            text_sample=text,
            clip_scores=shortlist, 
            filepath=filepath
        )

        if result:
            final_decision = result.get("final_domain", "unknown")
            justification = result.get("justification", "N/A")
            
            if final_decision in config.TARGET_DOMAINS or final_decision == "unknown":
                logger.info(f"Arbitrage rendu : {final_decision} ({justification})")
                return final_decision, ai_probabilities, "llm_multi_arbitrated"

    # Rejet si même le LLM ou la fusion échouent à donner une certitude
    if max_prob < 0.20 and best_final != "unknown":
        return "unknown", ai_probabilities, "low_confidence_rejection"

    return best_final, ai_probabilities, "semantic_fusion"