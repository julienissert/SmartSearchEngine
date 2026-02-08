# src/search/processor.py
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from src.embeddings.image_embeddings import embed_image
from src.utils.logger import setup_logger
from src.intelligence.llm_manager import get_llm
from src import config
logger = setup_logger("Processor")


try:
    logger.info(f"Chargement du modèle PaddleOCR (Langue: {config.OCR_LANG})...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=config.OCR_LANG, show_log=False)
    logger.info("PaddleOCR chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement de PaddleOCR : {e}")
    ocr_engine = None

def analyze_query(pil_image: Image.Image):
    """Prépare les vecteurs de recherche (fused et pure_visual)."""
    # 1. Encodage CLIP (Image -> Vector)
    vector = embed_image(pil_image)
    
    # 2. Extraction OCR
    ocr_text = ""
    if ocr_engine:
        img_array = np.array(pil_image)
        result = ocr_engine.ocr(img_array, cls=True)
        if result and result[0]:
            extracted_lines = [line[1][0] for line in result[0] if line[1][1] > 0.5]
            ocr_text = " ".join(extracted_lines).strip()
    
    # 3. Analyse d'intention via LLM
    if len(ocr_text) > 4:
        intent = get_llm().analyze_scan_intent(ocr_text)
    else:
        intent = {"domain": "unknown", "label": "unknown", "type": "image"}

    # --- ÉVOLUTION DOUBLE VECTEUR ---
    # Si on a du texte OCR, on génère aussi un vecteur texte pour fusionner
    from src.embeddings.text_embeddings import embed_text
    t_vec = embed_text(ocr_text) if ocr_text else None
    
    # Calcul du vecteur fusionné (Moyenne IA)
    vecs = [v for v in [t_vec, vector] if v is not None]
    fused_vector = np.mean(vecs, axis=0) if vecs else vector

    return {
        "fused_vector": fused_vector,     # Pour la recherche sémantique globale
        "pure_visual_vector": vector,    # Pour le match 100% image
        "ocr_text": ocr_text,
        "filters": intent
    }