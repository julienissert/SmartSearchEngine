# src/embeddings/text_embeddings.py
import torch
from transformers import CLIPTokenizer, CLIPModel 
from utils.preprocessing import clean_text
import config

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_text(text):
    clean = clean_text(text)
    
    inputs = tokenizer(
        text=[clean], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=77
    )
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    return text_features.squeeze().cpu().numpy()