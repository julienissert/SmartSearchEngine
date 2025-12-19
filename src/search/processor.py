# src/search/processor.py
import pytesseract
from PIL import Image
from embeddings.image_embeddings import embed_image
from embeddings.text_embeddings import embed_text

def analyze_query(pil_image: Image.Image):
    """Transforme l'image en vecteur et extrait le texte (OCR)"""
    
    vector = embed_image(pil_image)
    
    try:
        ocr_text = pytesseract.image_to_string(pil_image, lang="fra+eng").strip()
    except Exception:
        ocr_text = ""
        
    return vector, ocr_text