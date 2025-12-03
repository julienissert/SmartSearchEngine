# app/analyzer.py
from PIL import Image
from .embed import embed_image, embed_text
import pytesseract
from typing import List, Dict
import numpy as np

# Example list of class labels (you can expand to many Food101 labels)
# For zero-shot classification we'll compute text embeddings for each label and compare with image embedding
DEFAULT_FOOD_LABELS = [
    "Pizza Margherita",
    "Sushi",
    "Burger",
    "Salad",
    "Pasta Carbonara",
    "Ice cream",
    "Ramen",
    "Tacos",
    "Fried chicken",
    "French fries"
]

class ZeroShotClassifier:
    def __init__(self, labels: List[str] = None):
        self.labels = labels or DEFAULT_FOOD_LABELS
        # precompute label embeddings
        self.label_embs = np.vstack([embed_text(lbl) for lbl in self.labels]).astype("float32")

    def predict(self, image_emb: np.ndarray, top_k: int = 3):
        """
        image_emb assumed normalized. compute dot product with label_embs (cosine similarity).
        """
        sims = (self.label_embs @ image_emb).astype(float)  # shape (n_labels,)
        idx = np.argsort(-sims)[:top_k]
        return [{"label": self.labels[i], "score": float(sims[i])} for i in idx]

def ocr_extract(pil_image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(pil_image, lang="fra+eng")
        return text.strip()
    except Exception:
        return ""

# convenience: single analyzer function
_default_classifier = None

def analyze_image(pil_image: Image.Image, labels: List[str] = None):
    global _default_classifier
    if _default_classifier is None or (labels and labels != _default_classifier.labels):
        _default_classifier = ZeroShotClassifier(labels)

    img_emb = embed_image(pil_image)
    top_labels = _default_classifier.predict(img_emb, top_k=3)
    ocr_text = ocr_extract(pil_image)
    return {
        "image_embedding": img_emb,
        "labels": top_labels,
        "ocr_text": ocr_text
    }