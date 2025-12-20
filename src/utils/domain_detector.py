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

DOMAIN_VECTORS = {}
for domain in config.TARGET_DOMAINS:
    prompts = [template.format(domain) for template in PROMPT_TEMPLATES]
    embeddings = [embed_text(p) for p in prompts]
    DOMAIN_VECTORS[domain] = np.mean(embeddings, axis=0)

def detect_domain(text: str = None, pil_image: Image.Image = None):

    raw_scores = {}
    
    doc_emb = None
    if pil_image:
        doc_emb = embed_image(pil_image)
    elif text:
        doc_emb = embed_text(text)
    
    if doc_emb is None:
        return "unknown", {d: 0.0 for d in DOMAIN_VECTORS}

    for domain, domain_vec in DOMAIN_VECTORS.items():
        score = np.dot(doc_emb, domain_vec)
        raw_scores[domain] = float(score)

    total_score = sum(raw_scores.values())
    if total_score > 0:
        final_scores = {k: v / total_score for k, v in raw_scores.items()}
    else:
        final_scores = {k: 0.0 for k in raw_scores}

    best_domain = max(final_scores, key=final_scores.get)
    return best_domain, final_scores