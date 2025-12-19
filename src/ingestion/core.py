import os
from ingestion.dispatcher import dispatch_loader
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label

def process_document(doc, valid_labels):
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    
    domain, _ = detect_domain(text=str(content), pil_image=image)
    
    label = doc.get("suggested_label") or detect_label(
        filepath=source_path, text=str(content), image=image, known_labels=valid_labels
    )
    
    vector = embed_image(image) if image else embed_text(str(content))

    doc_id = store_metadata({
        "source": source_path,
        "type": doc["type"],
        "domain": domain,
        "label": label,
        "snippet": str(content)[:200] if content else ""
    })

    if vector is not None:
        add_to_index(domain, vector, doc_id)
    
    return domain 