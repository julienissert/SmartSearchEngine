# src/ingestion/core.py
import config
import numpy as np
from embeddings.text_embeddings import embed_text_batch
from embeddings.image_embeddings import embed_image_batch
from utils.domain_detector import detect_domain
from indexing.faiss_index import add_to_index
from indexing.metadata_index import store_metadata_batch 

def process_batch(batch_docs, valid_labels):
    if not batch_docs: return 0

    # 1. Vectorisation (GPU/CPU)
    texts = [str(d.get('content') or '') for d in batch_docs]
    images = [d.get('pil_image') for d in batch_docs]
    
    text_vectors = embed_text_batch(texts)
    
    image_vectors = [None] * len(batch_docs)
    valid_image_indices = [i for i, img in enumerate(images) if img is not None]
    
    if valid_image_indices:
        actual_images = [images[i] for i in valid_image_indices]
        actual_vectors = embed_image_batch(actual_images)
        for i, idx in enumerate(valid_image_indices):
            image_vectors[idx] = actual_vectors[i]

    # 2. Construction
    metadata_buffer = []
    
    for i, doc in enumerate(batch_docs):
        safe_content = str(doc.get('content') or '')
        doc['domain'] = detect_domain(safe_content, doc.get('pil_image'))
        
        vecs = []
        if text_vectors[i] is not None: vecs.append(text_vectors[i])
        if image_vectors[i] is not None: vecs.append(image_vectors[i])
        
        if not vecs:
            final_vector = np.zeros(config.EMBEDDING_DIM)
        else:
            final_vector = np.mean(vecs, axis=0)
        
        # Ajout FAISS
        faiss_id = add_to_index(final_vector, doc['domain'])
        
        # SÉCURITÉ : On ignore les documents "unknown" qui renvoient -1
        if faiss_id == -1:
            continue

        metadata_buffer.append({
            'id': int(faiss_id), # Conversion explicite en int Python
            'source': str(doc.get('source')),
            'file_hash': str(doc.get('file_hash')),
            'domain': str(doc.get('domain')),
            'content': safe_content[:5000],
            'extra': doc.get('extra', {})
        })

    # 3. Insertion SQL
    if metadata_buffer:
        store_metadata_batch(metadata_buffer)

    return len(metadata_buffer) # Retourne le nombre réel inséré