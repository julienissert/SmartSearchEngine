# src/embeddings/image_embeddings.py
import torch
from PIL import Image 
from transformers import CLIPImageProcessor, CLIPModel 
import config

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
image_processor = CLIPImageProcessor.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_image(img: Image.Image):
    
    inputs = image_processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        
    return emb.squeeze().cpu().numpy()