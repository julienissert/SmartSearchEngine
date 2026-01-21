# src/ingestion/core.py
from embeddings.text_embeddings import embed_text_batch
from embeddings.image_embeddings import embed_image_batch
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata_batch 
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label, resolve_structured_label

def process_batch(batch, valid_labels):

    if not batch: return 0

    # 1. Vectorisation massive (GPU / SIMD)
    images = [d.get("image") for d in batch if d.get("image") is not None]
    texts = [str(d.get("content", "")) for d in batch if d.get("image") is None]

    img_vectors = embed_image_batch(images) if images else []
    txt_vectors = embed_text_batch(texts) if texts else []

    # 2. Préparation (Calculs IA)
    prepared_data = []
    img_ptr, txt_ptr = 0, 0

    for doc in batch:
        vector = img_vectors[img_ptr] if doc.get("image") else txt_vectors[txt_ptr]
        if doc.get("image"): img_ptr += 1
        else: txt_ptr += 1
        
        meta, domain = _prepare_metadata(doc, vector, valid_labels)
        prepared_data.append({"meta": meta, "domain": domain, "vector": vector})

    # 3. SQL Batch avec Sécurité Intégrée
    sql_payload = [(d["meta"], d["domain"]) for d in prepared_data]
    local_ids = store_metadata_batch(sql_payload)

    indexed_count = 0
    for i, data in enumerate(prepared_data):
        doc_id = local_ids[i]
        
        if doc_id is not None:
            if data["vector"] is not None:
                add_to_index(data["domain"], data["vector"], doc_id)
                indexed_count += 1
            
    return indexed_count

def _prepare_metadata(doc, vector, valid_labels):
    """Logique métier de détection sans écriture disque."""
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    suggested = doc.get("suggested_label")
    file_hash = doc.get("file_hash")

    domain, ai_scores, _ = detect_domain(
        text=str(content), pil_image=image, 
        filepath=source_path,
        content_dict=content if isinstance(content, dict) else None,
        precomputed_vector=vector
    )
    score = ai_scores.get(domain, 0.0)

    if isinstance(content, dict):
        d_name = list(content.keys())[0] if doc.get("type") == "h5" else None
        label = resolve_structured_label(content, source_path, valid_labels, suggested, d_name)
    else:
        label = detect_label(source_path, str(content), image, valid_labels, suggested)

    metadata = {
        "source": source_path,
        "file_hash": file_hash,
        "type": doc["type"],
        "domain": domain,
        "label": label,
        "domain_score": round(score, 4),
        "raw_data": content if isinstance(content, dict) else None
    }

    if not isinstance(content, dict):
        text_content = str(content).strip()
        if text_content.lower() != label.lower():
            metadata["snippet"] = text_content[:500]

    return metadata, domain