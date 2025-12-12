# app/embed.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None

def _load():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def embed_image(pil_image: Image.Image) -> np.ndarray:
    """
    Return a 512-dim CLIP image embedding (float32, L2-normalized)
    """
    _load()
    inputs = _processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = _model.get_image_features(**inputs)
    emb = img_feats[0].cpu().numpy().astype("float32")
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def embed_text(text: str) -> np.ndarray:
    """
    Return a 512-dim CLIP text embedding (float32, L2-normalized)
    """
    _load()
    inputs = _processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        txt_feats = _model.get_text_features(**inputs)
    emb = txt_feats[0].cpu().numpy().astype("float32")
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb