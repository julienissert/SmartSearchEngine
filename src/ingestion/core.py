# src/ingestion/core.py
import os
import numpy as np
import gc
import torch
from src import config
from src.embeddings.text_embeddings import embed_text_batch
from src.embeddings.image_embeddings import embed_image_batch
from src.utils.domain_detector import detect_domain
from src.utils.label_detector import detect_label
from src.indexing.vector_store import add_documents, get_folder_contract, save_folder_contract 
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

    # --- ÉTAPE 2 : CERVEAU (Traitement Individuel & Préparation Buffers) ---
    metadata_buffer = []
    vector_buffer = []
    
    for i, doc in enumerate(batch_docs):
        try:
            # Fusion sémantique des vecteurs (Texte + Image)
            vecs = [v for v in [text_vectors[i], image_vectors[i]] if v is not None]
            final_vector = np.mean(vecs, axis=0) if vecs else np.zeros(config.EMBEDDING_DIM)
            
            # Analyse sémantique (Domaine & Label)
            meta, _ = _prepare_document_metadata(doc, final_vector, valid_labels)
            
            # Stockage dans les buffers de batch
            metadata_buffer.append(meta)
            vector_buffer.append(final_vector)
            
            # Libération précoce de la référence image dans le dictionnaire
            if 'image' in doc: doc['image'] = None 
                
        except Exception as e:
            logger.warning(f"Fichier ignoré car corrompu : {doc.get('source')} | Erreur: {e}")
            continue

    # --- ÉTAPE 3 : INSERTION UNIFIÉE DANS LE VECTOR STORE & NETTOYAGE ---
    indexed_count = 0
    if metadata_buffer:
        indexed_count = add_documents(metadata_buffer, vector_buffer)

    # 2. Libération CRITIQUE de la RAM (Objets PIL)
    for doc in batch_docs:
        if 'image' in doc and doc['image'] is not None:
            try:
                doc['image'].close() 
            except: pass
            doc['image'] = None
        doc.clear() 

    # 3. Purge complète des listes de support
    batch_docs.clear()
    metadata_buffer.clear()
    vector_buffer.clear()
    texts.clear()
    images.clear()
    gc.collect()
    
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()

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
            content_dict=raw_content if isinstance(raw_content, dict) else {}
        )
        domain = domain_res
        score = probs.get(domain, 0.0) if isinstance(probs, dict) else 0.0
        doc["extra"]["detection_method"] = method
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