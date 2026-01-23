# src/utils/domain_detector.py
import config
import numpy as np
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image

_domain_prompts_embeddings = None

def get_domain_embeddings():
    global _domain_prompts_embeddings
    if _domain_prompts_embeddings is None:
        prompts = [
            "medical document, health, doctor, prescription",
            "food document, recipe, nutrition, restaurant"
        ]
        # Ceci va trigger le chargement GPU (uniquement dans le MainProcess)
        _domain_prompts_embeddings = [embed_text(p) for p in prompts]
    return _domain_prompts_embeddings

def detect_domain(text, pil_image=None):
    # Sécurité type
    safe_text = str(text) if text else ""
    
    domain_vectors = get_domain_embeddings()
    doc_vectors = []
    
    # Texte significatif uniquement
    if len(safe_text.strip()) > 5:
        doc_vectors.append(embed_text(safe_text))
        
    if pil_image is not None:
        doc_vectors.append(embed_image(pil_image))
    
    if not doc_vectors: return "unknown"

    final_doc_vector = np.mean(doc_vectors, axis=0)
    
    similarities = []
    for dv in domain_vectors:
        # Sécurité division par zéro
        norm_product = (np.linalg.norm(final_doc_vector) * np.linalg.norm(dv))
        if norm_product == 0:
            similarities.append(0)
        else:
            sim = np.dot(final_doc_vector, dv) / norm_product
            similarities.append(sim)

    best_idx = np.argmax(similarities)
    if similarities[best_idx] < config.SEMANTIC_THRESHOLD:
        return "unknown"
    return config.TARGET_DOMAINS[best_idx]