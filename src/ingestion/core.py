# src/ingestion/core.py
import os
import numpy as np
import gc
import torch
import time
import psutil
from src import config
from PIL import Image
from src.embeddings.text_embeddings import embed_text_batch
from src.embeddings.image_embeddings import embed_image_batch
from src.intelligence.domain_detector import detect_domain
from src.intelligence.label_detector import detect_label
from src.indexing.vector_store import add_documents, get_folder_contract, save_folder_contract
from src.utils.logger import setup_logger
from src.intelligence.llm_manager import llm  
from src.ingestion.dispatcher import is_visual_type

_BATCH_COUNTER = 0
logger = setup_logger("IngestionCore")
_SESSION_IA_CACHE = {}

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

    global _BATCH_COUNTER
    if not batch_docs: 
        return 0, "unknown", 0.0

    _BATCH_COUNTER += 1

    # --- ÉTAPE 1 : CHARGEMENT JIT (Just In Time) ---
    actual_images = []
    for d in batch_docs:
        img = None
        source = str(d.get('source', ''))
        
        if is_visual_type(source):
            try:
                img = Image.open(source).convert('RGB')
            except Exception:
                img = None
        actual_images.append(img)

    # --- ÉTAPE 2 : VECTORISATION BATCH ---
    texts = [str(d.get('content') or '') for d in batch_docs]
    
    try:
        # Vectorisation Texte Batch 
        text_vectors = embed_text_batch(texts)
        
        # Vectorisation Image Batch 
        image_vectors = [None] * len(batch_docs)
        valid_img_idx = [i for i, img in enumerate(actual_images) if img is not None]
        
        if valid_img_idx:
            actual_vecs = embed_image_batch([actual_images[i] for i in valid_img_idx])
            for i, idx in enumerate(valid_img_idx):
                image_vectors[idx] = actual_vecs[i]
                
    except Exception as e:
        logger.error(f"Erreur fatale lors de la vectorisation du batch : {e}")
        # Sécurité : on ferme les images ouvertes avant de quitter
        for img in actual_images:
            if img: img.close()
        return 0, "unknown", 0.0

    # --- ÉTAPE 3 : CERVEAU & PRÉPARATION MÉTADONNÉES ---
    metadata_buffer = []
    vector_buffer = []
    last_domain = "unknown"
    last_score = 0.0
    
    for i, doc in enumerate(batch_docs):
        try:
            vecs = [v for v in [text_vectors[i], image_vectors[i]] if v is not None]
            final_vector = np.mean(vecs, axis=0) if vecs else np.zeros(config.EMBEDDING_DIM)
            
            doc['image'] = actual_images[i] 
            
            # Analyse IA (Domaine, Label, Score) - Récupération du triplet
            meta, domain, score = _prepare_document_metadata(doc, final_vector, image_vectors[i], valid_labels)
            
            if domain != "unknown":
                last_domain = domain
                last_score = score

            metadata_buffer.append(meta)
            vector_buffer.append(final_vector)
            
            # --- NETTOYAGE PHYSIQUE IMMÉDIAT ---
            if actual_images[i]:
                actual_images[i].close() 
            doc['image'] = None 
                
        except Exception as e:
            logger.warning(f"Fichier ignoré : {doc.get('source')} | {e}")
            continue
        
    # --- ÉTAPE 4 : INSERTION ---
    indexed_count = 0
    if metadata_buffer:
        indexed_count = add_documents(metadata_buffer, vector_buffer)
        
    # --- ÉTAPE 5 : NETTOYAGE  (MODULO) ---
    if _BATCH_COUNTER % config.CLEANUP_MODULO == 0:
        gc.collect()
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Purge finale des listes temporaires
    batch_docs.clear()
    actual_images.clear()
    metadata_buffer.clear()
    vector_buffer.clear()

    return indexed_count, last_domain, last_score

def _prepare_document_metadata(doc, vector, img_vector, valid_labels):
   
    source_path = doc["source"]
    raw_content = doc.get("content") 
    content_str = str(raw_content or "")
    
    archive_path = _get_archive_entity(source_path)
    contract = get_folder_contract(archive_path)
    
    c_domain = contract.get("assigned_domain") if contract else None
    c_score = contract.get("confidence") if contract else 0.0
    
    if c_domain and c_domain != "unknown":
        domain = c_domain
        actual_score = c_score
        score = 1.0 
        method = "contract_trust"
        
    elif archive_path in _SESSION_IA_CACHE:
        cache = _SESSION_IA_CACHE[archive_path]
        domain = cache['domain']
        actual_score = cache['score']
        score = 1.0 # Validation par session
        method = "session_cache" 
           
    else:
        domain_res, probs, detect_method = detect_domain(
            filepath=source_path, 
            precomputed_vector=vector,
            content_dict=raw_content if isinstance(raw_content, dict) else {}
        )
        domain = domain_res
        actual_score = probs.get(domain, 0.0) if isinstance(probs, dict) else 0.0
        method = detect_method

        # Arbitrage LLM si CLIP est incertain ET Ollama est dispo
        # (Le timeout est géré par la config dans llm_manager)
        if actual_score < 0.8 and llm.is_healthy():
            logger.info(f"Arbitrage LLM requis pour : {os.path.basename(source_path)}")
            res = llm.arbitrate_domain(content_str, probs, source_path)
            
            domain = res.get("final_domain", domain)
            actual_score = float(res.get("confidence", actual_score))
            method = "llm_arbitration"
            doc["extra"]["llm_justification"] = res.get("justification")
            
            # --- CRUCIAL : On mémorise en RAM pour les 1000 prochaines lignes du CSV ---
            _SESSION_IA_CACHE[archive_path] = {"domain": domain, "score": actual_score}
        
        score = actual_score

    # 2. Résolution du Label (Intelligence niveau 3)
    label = detect_label(
        filepath=source_path, 
        content=raw_content, 
        image=doc.get('image'),
        image_vector=img_vector, 
        label_mapping=valid_labels,
        type=doc.get('type')
    )

    # 3. Métadonnées finales
    metadata = {
        "source": source_path,
        "file_hash": doc.get("file_hash"),
        "type": doc.get("type", "unknown"),
        "domain": str(domain),
        "label": label,
        "domain_score": round(float(score), 4),
        "content": content_str[:20000], 
        "snippet": content_str[:500],   
        "extra": {
            **doc.get("extra", {}), 
            "detection_method": method,
            "ingested_at": time.time(),
            "ram_usage": f"{psutil.virtual_memory().percent}%"
        }
    }

    return metadata, domain, actual_score