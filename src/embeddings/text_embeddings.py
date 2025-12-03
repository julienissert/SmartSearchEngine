# src/embeddings/text_embeddings.py
import torch
from transformers import CLIPProcessor, CLIPModel
from utils.preprocessing import clean_text
import config

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_text(text):
    clean = clean_text(text)
    
    # Troncature à 77 tokens (limite technique de CLIP)
    inputs = processor(text=[clean], return_tensors="pt", padding=True, truncation=True, max_length=77)
    
    with torch.no_grad():
        # Renvoie un vecteur de taille 512 compatible avec l'image
        text_features = model.get_text_features(**inputs)
        
    return text_features.squeeze().cpu().numpy()