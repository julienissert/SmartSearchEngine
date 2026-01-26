# src/embeddings/text_embeddings.py
import torch
import multiprocessing
import numpy as np
from transformers import CLIPModel, CLIPTokenizer
from src import config
from src.utils.preprocessing import clean_text

_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        current_proc = multiprocessing.current_process().name
        is_worker = any(x in current_proc for x in ["Process-", "ForkPoolWorker", "engine_ingest"])
        device = "cpu" if is_worker else config.DEVICE
        
        _model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(device)
        _tokenizer = CLIPTokenizer.from_pretrained(config.IMAGE_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer

def embed_text_batch(texts):
    if not texts: return []

    cleans = [clean_text(str(t)) if t is not None else "" for t in texts]
    
    model, tokenizer = get_model()
    if model is None or tokenizer is None:
        return []
    
    inputs = tokenizer(
        cleans, 
        padding=True, 
        truncation=True, 
        max_length=77, 
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    
    # Normalisation L2 
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.cpu().numpy()

def embed_text(text):
    if not text: return np.zeros(config.EMBEDDING_DIM)
    return embed_text_batch([text])[0]