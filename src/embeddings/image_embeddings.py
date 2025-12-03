#embeddings/image_embeddings.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import config

model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL_NAME)

def embed_image(img):
    inputs = processor(images=img, return_tensors="pt")    
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.squeeze().cpu().numpy()
