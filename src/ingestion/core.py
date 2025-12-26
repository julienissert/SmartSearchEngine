# src/ingestion/core.py
from embeddings.text_embeddings import embed_text
from embeddings.image_embeddings import embed_image
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata
from utils.domain_detector import detect_domain
from utils.label_detector import detect_label, resolve_structured_label

def process_document(doc, valid_labels):
    """
    Orchestre le pipeline de traitement d'un document :
    Domaine -> Label (Stratégie Hybride) -> Vecteur -> Indexation.
    """
    content = doc.get("content", "")
    image = doc.get("image", None)
    source_path = doc["source"]
    suggested = doc.get("suggested_label") 
    
    # 1. Détection du domaine (Food / Medical)
    domain, domain_scores = detect_domain(text=str(content), pil_image=image)
    score = domain_scores.get(domain, 0.0)    

    # 2. DÉTECTION DU LABEL (Délégation totale au détecteur centralisé)
    if isinstance(content, dict):
        # Pour le H5, on extrait le nom du dataset pour la clé de cache composite
        d_name = list(content.keys())[0] if doc.get("type") == "h5" else None
        
        # On passe 'suggested' pour que le détecteur gère la priorité Couche 0
        label = resolve_structured_label(
            content, 
            source_path, 
            label_mapping=valid_labels, 
            suggested_label=suggested,
            dataset_name=d_name
        )
    else:
        # Pour Image, PDF, TXT : détection par contenu avec Couche 0 prioritaire
        label = detect_label(
            filepath=source_path, 
            text=str(content), 
            image=image, 
            label_mapping=valid_labels,
            suggested_label=suggested
        )
    
    # 3. Calcul de l'embedding (Vectorisation CLIP)
    vector = embed_image(image) if image else embed_text(str(content))

    # 4. Stockage des métadonnées enrichies
    doc_id = store_metadata({
            "source": source_path,
            "type": doc["type"],
            "domain": domain,
            "label": label,
            "domain_score": round(score, 4),  
            # Snippet adapté selon si la donnée est structurée ou non
            "snippet": str(content)[:200] if isinstance(content, str) else str(list(content.values())[:3])
        }, domain)

    # 5. Ajout à l'index FAISS spécifique au domaine
    if vector is not None:
        add_to_index(domain, vector, doc_id)
    
    return domain