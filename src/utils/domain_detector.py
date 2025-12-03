# utils/domain_detector.py
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image
import numpy as np
from PIL import Image

DOMAINS = ["food", "medical", "unknown"]

def detect_domain(text: str = None, pil_image: Image.Image = None,
                  text_weight: float = 0.6, image_weight: float = 0.4):

    scores = np.zeros(len(DOMAINS))
    
    if text:
        text_emb = embed_text(text)
        domain_embs = np.vstack([embed_text(d) for d in DOMAINS])
        text_scores = domain_embs @ text_emb
        if np.linalg.norm(text_scores) > 0:
            text_scores = text_scores / np.linalg.norm(text_scores)
        scores += text_weight * text_scores
    
    if pil_image:
        img_emb = embed_image(pil_image)
        domain_embs = np.vstack([embed_text(d) for d in DOMAINS])
        img_scores = domain_embs @ img_emb
        if np.linalg.norm(img_scores) > 0:
            img_scores = img_scores / np.linalg.norm(img_scores)
        scores += image_weight * img_scores
    
    if scores.sum() > 0:
        scores = scores / scores.sum()
    
    domain_confidence = {d: float(scores[i]) for i, d in enumerate(DOMAINS)}
    best_domain = max(domain_confidence, key=domain_confidence.get)
    
    return best_domain, domain_confidence
