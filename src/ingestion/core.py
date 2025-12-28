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

    # 2. Vectorisation massive (Un seul appel au GPU/CPU par type)
    img_vectors = embed_image_batch(images) if images else []
    txt_vectors = embed_text_batch(texts) if texts else []

    # 3. Finalisation individuelle (Domaine, Label, Stockage)
    img_ptr, txt_ptr = 0, 0
    for doc in batch:
        if doc.get("image"):
            vector = img_vectors[img_ptr]
            img_ptr += 1
        else:
            vector = txt_vectors[txt_ptr]
            txt_ptr += 1
            
        # Enregistrement des métadonnées et indexation FAISS
        _finalize_single_doc(doc, vector, valid_labels)

def _finalize_single_doc(doc, vector, valid_labels):
    """Gère la logique métier et le stockage pour un document unique."""
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    suggested = doc.get("suggested_label")

    # Détection du domaine
    domain, ai_scores, _ = detect_domain(
        text=str(content), pil_image=image, 
        filepath=source_path,
        content_dict=content if isinstance(content, dict) else None,
        precomputed_vector=vector
    )
    score = ai_scores.get(domain, 0.0)

    # Détection du label
    if isinstance(content, dict):
        d_name = list(content.keys())[0] if doc.get("type") == "h5" else None
        label = resolve_structured_label(content, source_path, valid_labels, suggested, d_name)
    else:
        label = detect_label(source_path, str(content), image, valid_labels, suggested)

    # Préparation des métadonnées
    metadata = {
        "source": source_path,
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

    # Persistance
    doc_id = store_metadata(metadata, domain)
    if vector is not None:
        add_to_index(domain, vector, doc_id)