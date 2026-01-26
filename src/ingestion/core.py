# src/ingestion/core.py
import os
import numpy as np
from src import config
from src.embeddings.text_embeddings import embed_text_batch
from src.embeddings.image_embeddings import embed_image_batch
from src.utils.domain_detector import detect_domain
from src.indexing.faiss_index import add_to_index
from src.indexing.metadata_index import store_metadata_batch, get_folder_contract, save_folder_contract
from src.utils.label_detector import detect_label 
from src.utils.logger import setup_logger

logger = setup_logger("IngestionCore")

def _get_archive_entity(source_path):
    try:
        rel = os.path.relpath(source_path, config.DATASET_DIR)
        parts = rel.split(os.sep)
        if len(parts) > 1:
            return os.path.join(config.DATASET_DIR, parts[0])
    except Exception:
        pass
    return os.path.dirname(source_path)

def process_batch(batch_docs, valid_labels):
    if not batch_docs: return 0

    # --- ÉTAPE 1 : MUSCLES (Vectorisation Batch) ---
    texts = [str(d.get('content') or '') for d in batch_docs]
    images = [d.get('image') for d in batch_docs]
    
    try:
        text_vectors = embed_text_batch(texts)        
        image_vectors = [None] * len(batch_docs)
        valid_img_idx = [i for i, img in enumerate(images) if img is not None]
        if valid_img_idx:
            actual_vecs = embed_image_batch([images[i] for i in valid_img_idx])
            for i, idx in enumerate(valid_img_idx):
                image_vectors[idx] = actual_vecs[i]
    except Exception as e:
        logger.error(f"Erreur fatale lors de la vectorisation du batch : {e}")
        return 0

    # --- ÉTAPE 2 : CERVEAU (Traitement Individuel Résilient) ---
    metadata_buffer = []
    
    for i, doc in enumerate(batch_docs):
        try:
            # Fusion des vecteurs (Moyenne sémantique)
            vecs = [v for v in [text_vectors[i], image_vectors[i]] if v is not None]
            final_vector = np.mean(vecs, axis=0) if vecs else np.zeros(config.EMBEDDING_DIM)
            
            # Préparation des métadonnées
            meta, domain = _prepare_document_metadata(doc, final_vector, valid_labels)
            
            # Libération immédiate de la RAM
            if 'image' in doc: del doc['image']
            
            # Synchronisation FAISS
            local_id = add_to_index(final_vector, domain)
            
            if local_id != -1:
                meta['local_id'] = local_id
                metadata_buffer.append(meta)
                
        except Exception as e:
            logger.warning(f"Fichier ignoré car corrompu : {doc.get('source')} | Erreur: {e}")
            continue

    # --- ÉTAPE 3 : SQL Batch & NETTOYAGE AGRESSIF ---
    if metadata_buffer:
        store_metadata_batch(metadata_buffer)

    # 1. Capturer le nombre RÉEL d'insertions AVANT le nettoyage
    indexed_count = len(metadata_buffer)

    # 2. Libération CRITIQUE de la RAM (PIL Images)
    # On ferme explicitement chaque image pour libérer les pixels bruts
    for doc in batch_docs:
        if 'image' in doc and doc['image'] is not None:
            try:
                doc['image'].close() 
            except: pass
            doc['image'] = None

    # 3. Nettoyage des listes de support
    batch_docs.clear()
    metadata_buffer.clear()
    texts.clear()
    images.clear()
    
    # 4. Forcer le Garbage Collector de Python
    import gc
    gc.collect()
    
    # 5. Vidage du cache CUDA (si GPU)
    if config.DEVICE == "cuda":
        import torch
        torch.cuda.empty_cache()

    # On renvoie le vrai nombre d'insertions (pour tes logs)
    return indexed_count

def _prepare_document_metadata(doc, vector, valid_labels):
    source_path = doc["source"]
    raw_content = doc.get("content") 
    content_str = str(raw_content or "")
    
    # 1. Vérification du Contrat de l'Archive(n)
    archive_path = _get_archive_entity(source_path)
    domain = get_folder_contract(archive_path)
    score = 1.0
    
    if not domain:
        domain_res, probs, method = detect_domain(
            filepath=source_path, 
            precomputed_vector=vector,
            content_dict=raw_content if isinstance(raw_content, dict) else None
        )
        domain = domain_res
        score = probs.get(domain, 0.0) if isinstance(probs, dict) else 0.0
        
        # SÉCURITÉ 
        if score > 0.7:
            save_folder_contract(archive_path, domain, score, verified=0)

    # 2. Résolution du Label 
    label = detect_label(
        filepath=source_path, 
        content=raw_content, 
        image=doc.get('image'), 
        label_mapping=valid_labels,
        type=doc.get('type')
    )

    # 3. Métadonnées 
    metadata = {
        "source": source_path,
        "file_hash": doc.get("file_hash"),
        "type": doc.get("type", "unknown"),
        "domain": domain,
        "label": label,
        "domain_score": round(float(score), 4),
        "content": content_str[:20000], 
        "snippet": content_str[:500],   
        "extra": doc.get("extra", {})
    }

    return metadata, domain