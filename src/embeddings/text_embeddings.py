# src/embeddings/text_embeddings.py
import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPModel 
from utils.preprocessing import clean_text
import config 

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(config.DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_text(text: str):
    return embed_text_batch([text])[0]

def embed_text_batch(texts: list, micro_batch_size: int = None):
    if not texts: 
        return []
    
    if micro_batch_size is None:
        micro_batch_size = getattr(config, 'CLIP_TEXT_MICRO_BATCH', 128)
        
    all_features = []
    
    for i in range(0, len(texts), micro_batch_size):
        chunk = texts[i : i + micro_batch_size]
        cleans = [clean_text(t) for t in chunk] 
        
        inputs = tokenizer(
            text=cleans, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(config.DEVICE)
        
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            all_features.append(features.cpu().numpy()) 
            
    return np.concatenate(all_features, axis=0)