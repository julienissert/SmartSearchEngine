# src/utils/domain_detector.py
import numpy as np
from PIL import Image
import config  
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

PROMPT_TEMPLATES = [
    "A photo of {}.",
    "A text about {}.",
    "Category: {}.",
    "This is related to {}."
]

# normalisation des vecteurs
DOMAIN_VECTORS = {}
for domain in config.TARGET_DOMAINS:
    prompts = [template.format(domain) for template in PROMPT_TEMPLATES]
    embeddings = [embed_text(p) for p in prompts]
    
    vec = np.mean(embeddings, axis=0)
    
    norm = np.linalg.norm(vec)
    DOMAIN_VECTORS[domain] = vec / norm if norm > 0 else vec

def detect_domain(text: str = None, pil_image: Image.Image = None):
    raw_scores = {}
    
    # 1. Extraction de l'embedding du document
    doc_emb = None
    if pil_image:
        doc_emb = embed_image(pil_image)
    elif text:
        doc_emb = embed_text(text)
    
    if doc_emb is None:
        return "unknown", {d: 0.0 for d in DOMAIN_VECTORS}

    # Normalisation de l'embedding du document pour le calcul cosinus
    doc_norm = np.linalg.norm(doc_emb)
    if doc_norm > 0:
        doc_emb = doc_emb / doc_norm

    # 2. Calcul des similarités cosinus 
    for domain, domain_vec in DOMAIN_VECTORS.items():
        score = np.dot(doc_emb, domain_vec)
        raw_scores[domain] = float(score)

    # 3. Logit Scaling (Standard OpenAI CLIP)
    # de température logit (100.0) pour transformer la similarité en probabilité.
    logit_scale = 100.0 
    scores_array = np.array(list(raw_scores.values()))
    logits = scores_array * logit_scale
    
    # 4. Softmax stable pour la distribution des probabilités
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)
    
    final_scores = {k: float(p) for k, p in zip(raw_scores.keys(), probabilities)}

    best_domain = max(final_scores, key=final_scores.get)
    return best_domain, final_scores