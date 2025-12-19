import numpy as np
from PIL import Image
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

TARGET_DOMAINS = ["food", "medical"]

PROMPT_TEMPLATES = [
    "A photo of {}.",
    "A text about {}.",
    "Category: {}.",
    "This is related to {}."
]

DOMAIN_VECTORS = {}
for domain in TARGET_DOMAINS:
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

    try:
        values = np.array(list(raw_scores.values()))
        keys = list(raw_scores.keys())
        
        max_val = np.max(values)
        exp_values = np.exp((values - max_val) * 30) 
        
        total = np.sum(exp_values)
        probabilities = exp_values / total
        
        final_scores = {k: float(p) for k, p in zip(keys, probabilities)}

    except Exception:
        final_scores = {d: 0.0 for d in raw_scores}

    best_domain = max(final_scores, key=final_scores.get)
    return best_domain, final_scores