# src/embeddings/image_embeddings.py
import torch
from PIL import Image 
from transformers import CLIPImageProcessor, CLIPModel 
import config

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
model.to(config.DEVICE) 

image_processor = CLIPImageProcessor.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_image(img: Image.Image):
    inputs = image_processor(images=img, return_tensors="pt").to(config.DEVICE)
    
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        
    return emb.squeeze().cpu().numpy()

def embed_image_batch(images: list):
    if not images: return []
    inputs = image_processor(images=images, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()