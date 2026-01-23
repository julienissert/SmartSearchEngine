# src/ingestion/core.py
from src import config
import numpy as np
from src.embeddings.text_embeddings import embed_text_batch
from src.embeddings.image_embeddings import embed_image_batch
from src.utils.domain_detector import detect_domain
from src.indexing.faiss_index import add_to_index
from src.indexing.metadata_index import store_metadata_batch 

def process_batch(batch_docs, valid_labels):
    if not batch_docs: return 0

    # 1. Vectorisation massive (GPU/CPU)
    texts = [str(d.get('content') or '') for d in batch_docs]
    
    # MODIFICATION 1 : Utilisation de la clé 'image' (cohérent avec ImageLoader)
    images = [d.get('image') for d in batch_docs] 
    
    text_vectors = embed_text_batch(texts)
    
    image_vectors = [None] * len(batch_docs)
    valid_image_indices = [i for i, img in enumerate(images) if img is not None]
    
    if valid_image_indices:
        actual_images = [images[i] for i in valid_image_indices]
        actual_vectors = embed_image_batch(actual_images)
        for i, idx in enumerate(valid_image_indices):
            image_vectors[idx] = actual_vectors[i]

    # 2. Construction et Nettoyage
    metadata_buffer = []
    
    for i, doc in enumerate(batch_docs):
        safe_content = str(doc.get('content') or '')
        
        # MODIFICATION 2 : Calcul du vecteur AVANT la détection de domaine
        vecs = []
        if text_vectors[i] is not None: vecs.append(text_vectors[i])
        if image_vectors[i] is not None: vecs.append(image_vectors[i])
        
        if not vecs:
            final_vector = np.zeros(config.EMBEDDING_DIM)
        else:
            final_vector = np.mean(vecs, axis=0)
            
        # MODIFICATION 3 : Suppression du double calcul (on passe precomputed_vector)
        # On passe None pour pil_image car le travail est déjà fait dans final_vector
        doc['domain'] = detect_domain(safe_content, None, precomputed_vector=final_vector)
        
        # MODIFICATION 4 : Libération CRITIQUE de la RAM
        # On supprime l'image PIL lourde immédiatement après vectorisation
        if 'image' in doc:
            del doc['image']
        
        # Ajout FAISS
        faiss_id = add_to_index(final_vector, doc['domain'])
        
        if faiss_id == -1:
            continue

        metadata_buffer.append({
            'id': int(faiss_id),
            'source': str(doc.get('source')),
            'file_hash': str(doc.get('file_hash')),
            'domain': str(doc.get('domain')),
            'content': safe_content[:5000],
            'extra': doc.get('extra', {})
        })

    # 3. Insertion SQL
    if metadata_buffer:
        store_metadata_batch(metadata_buffer)

    return len(metadata_buffer)