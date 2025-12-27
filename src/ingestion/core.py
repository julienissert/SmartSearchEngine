# src/ingestion/core.py
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label, resolve_structured_label

def process_document(doc, valid_labels):
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    suggested = doc.get("suggested_label") 
    
    # 1. Détection du domaine (IA scores calculés en arrière-plan)
    domain, ai_scores, method = detect_domain(
        text=str(content), pil_image=image, 
        filepath=source_path,
        content_dict=content if isinstance(content, dict) else None
    )
    
    # On récupère le score que l'IA attribue au domaine choisi
    # Si dossier=food et IA pense medical, ce score sera très proche de 0
    score = ai_scores.get(domain, 0.0)    

    # 2. Détection du label (Dossier parent prioritaire aussi ici)
    if isinstance(content, dict):
        d_name = list(content.keys())[0] if doc.get("type") == "h5" else None
        label = resolve_structured_label(content, source_path, label_mapping=valid_labels, suggested_label=suggested, dataset_name=d_name)
    else:
        label = detect_label(filepath=source_path, text=str(content), image=image, label_mapping=valid_labels, suggested_label=suggested)
    
    # 3. Vectorisation
    vector = embed_image(image) if image else embed_text(str(content))

    # 4. Métadonnées enrichies
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

    doc_id = store_metadata(metadata, domain)
    if vector is not None:
        add_to_index(domain, vector, doc_id)
    
    return domain