# src/ingestion/core.py
from embeddings.text_embeddings import embed_text_batch
from embeddings.image_embeddings import embed_image_batch
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label, resolve_structured_label

def process_batch(batch, valid_labels):
    """
    Orchestre le traitement d'un lot de documents.
    Optimise les performances en groupant les appels au modèle CLIP.
    """
    # 1. Séparation des types pour la vectorisation groupée
    images = [d.get("image") for d in batch if d.get("image") is not None]
    texts = [str(d.get("content", "")) for d in batch if d.get("image") is None]

    # 2. Vectorisation massive
    img_vectors = embed_image_batch(images) if images else []
    txt_vectors = embed_text_batch(texts) if texts else []

    # 3. Finalisation individuelle
    indexed_count = 0
    img_ptr, txt_ptr = 0, 0
    for doc in batch:
        if doc.get("image"):
            vector = img_vectors[img_ptr]
            img_ptr += 1
        else:
            vector = txt_vectors[txt_ptr]
            txt_ptr += 1
            
        # On passe le hash calculé dans le service
        if _finalize_single_doc(doc, vector, valid_labels):
            indexed_count += 1
            
    return indexed_count

def _finalize_single_doc(doc, vector, valid_labels):
    """Gère la logique métier et le stockage pour un document unique."""
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    suggested = doc.get("suggested_label")
    file_hash = doc.get("file_hash") # Hash récupéré du service

    # ON GARDE TON INTELLIGENCE ICI : Détection domaine
    domain, ai_scores, _ = detect_domain(
        text=str(content), pil_image=image, 
        filepath=source_path,
        content_dict=content if isinstance(content, dict) else None,
        precomputed_vector=vector
    )
    score = ai_scores.get(domain, 0.0)

    # ON GARDE TON INTELLIGENCE ICI : Détection label
    if isinstance(content, dict):
        d_name = list(content.keys())[0] if doc.get("type") == "h5" else None
        label = resolve_structured_label(content, source_path, valid_labels, suggested, d_name)
    else:
        label = detect_label(source_path, str(content), image, valid_labels, suggested)

    # Préparation des métadonnées (incluant le hash)
    metadata = {
        "source": source_path,
        "file_hash": file_hash, # Sécurité doublon
        "type": doc["type"],
        "domain": domain,
        "label": label,
        "domain_score": round(score, 4)
    }

    if isinstance(content, dict):
        metadata["raw_data"] = content
    else:
        text_content = str(content).strip()
        if text_content.lower() != label.lower():
            metadata["snippet"] = text_content[:500]

    # Persistance (store_metadata renvoie None si doublon de hash)
    doc_id = store_metadata(metadata, domain)
    
    if doc_id is not None:
        # Seulement si c'est un nouveau document, on l'ajoute à FAISS
        if vector is not None:
            add_to_index(domain, vector, doc_id)
        return True
    
    return False