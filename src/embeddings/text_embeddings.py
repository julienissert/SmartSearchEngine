# src/embeddings/text_embeddings.py
import torch
import multiprocessing
import config
import numpy as np
from transformers import CLIPModel, CLIPTokenizer

_model = None
_tokenizer = None

def get_model():
    """Charge le modèle (CPU pour workers, GPU pour Main)."""
    global _model, _tokenizer
    if _model is None:
        current_proc = multiprocessing.current_process().name
        # On force le CPU pour les workers (évite le blocage 0%)
        is_worker = any(x in current_proc for x in ["Process-", "ForkPoolWorker", "engine_ingest"])
        
        device = "cpu" if is_worker else config.DEVICE
        
        print(f"🔄 Chargement CLIP TEXTE sur {device} (Process: {current_proc})")
        
        _model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(device)
        _tokenizer = CLIPTokenizer.from_pretrained(config.IMAGE_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer

def embed_text_batch(texts):
    """Vectorisation robuste : convertit tout en string."""
    if not texts: return []
    
    # --- SECURITE ABSOLUE ---
    # Convertit None -> "" et force le type str pour tout le reste
    # C'est cette ligne qui corrige votre erreur 'text input must be str'
    clean_texts = [str(t) if t is not None else "" for t in texts]
    
    model, tokenizer = get_model()
    device = model.device
    
    inputs = tokenizer(clean_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def embed_text(text):
    if not text: return np.zeros(config.EMBEDDING_DIM)
    return embed_text_batch([text])[0]