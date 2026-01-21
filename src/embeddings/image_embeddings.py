# src/embeddings/image_embeddings.py
import torch
import numpy as np
from PIL import Image 
from transformers import CLIPImageProcessor, CLIPModel 
import config 

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME).to(config.DEVICE)
image_processor = CLIPImageProcessor.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_image(img: Image.Image):
    return embed_image_batch([img])[0] 

def embed_image_batch(images: list, micro_batch_size: int = None):
    if not images: 
        return []

    if micro_batch_size is None:
        micro_batch_size = getattr(config, 'CLIP_IMAGE_MICRO_BATCH', 32)
        
    all_features = []
    
    for i in range(0, len(images), micro_batch_size):
        chunk = images[i : i + micro_batch_size]
        
        inputs = image_processor(images=chunk, return_tensors="pt", padding=True).to(config.DEVICE)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            all_features.append(features.cpu().numpy()) 
            
    return np.concatenate(all_features, axis=0)