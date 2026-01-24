# src/embeddings/image_embeddings.py
import torch
import multiprocessing
from src import config
import numpy as np
from transformers import CLIPProcessor, CLIPModel

_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        current_proc = multiprocessing.current_process().name
        is_worker = any(x in current_proc for x in ["Process-", "ForkPoolWorker", "engine_ingest"])
        device = "cpu" if is_worker else config.DEVICE
                
        _model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(device)
        _processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL_NAME)
        _model.eval()
    return _model, _processor

def embed_image_batch(pil_images):
    # Filtrage : On retire les None avant d'appeler le modèle
    valid_imgs = [img for img in pil_images if img is not None]
    if not valid_imgs: return []

    model, processor = get_model()
    inputs = processor(images=valid_imgs, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def embed_image(pil_image):
    if pil_image is None:
        return np.zeros(config.EMBEDDING_DIM)
    res = embed_image_batch([pil_image])
    return res[0] if len(res) > 0 else np.zeros(config.EMBEDDING_DIM)