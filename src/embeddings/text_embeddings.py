# src/embeddings/text_embeddings.py
import torch
from transformers import CLIPTokenizer, CLIPModel 
from utils.preprocessing import clean_text
import config

# Chargement séparé pour éviter l'erreur Pylance
model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
model.to(config.DEVICE)

tokenizer = CLIPTokenizer.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_text(text: str):
    clean = clean_text(text)
    inputs = tokenizer(
        text=[clean], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=77
    ).to(config.DEVICE)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    return text_features.squeeze().cpu().numpy()

def embed_text_batch(texts: list):
    if not texts: return []
    cleans = [clean_text(t) for t in texts]
    inputs = tokenizer(text=cleans, return_tensors="pt", padding=True, truncation=True, max_length=77).to(config.DEVICE)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.cpu().numpy()