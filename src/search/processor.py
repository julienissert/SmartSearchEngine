# src/search/processor.py
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from src.embeddings.image_embeddings import embed_image
from src.utils.logger import setup_logger
from src.intelligence.llm_manager import llm
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
    """
    Transforme l'image en vecteur (CLIP) et extrait le texte (PaddleOCR).
    """
    
    # 1. Encodage vectoriel (Image -> Vector)
    vector = embed_image(pil_image)
    
    # 2. Extraction OCR (Image -> Text)
    ocr_text = ""
    
    if ocr_engine:
        try:
            # Conversion PIL (RGB) -> Numpy Array pour Paddle
            img_array = np.array(pil_image)            
            result = ocr_engine.ocr(img_array, cls=True)
            extracted_lines = []
            if result and result[0]:
                for line in result[0]:
                    text_detected = line[1][0]
                    confidence = line[1][1]                   
                    if confidence > 0.5: 
                        extracted_lines.append(text_detected)           
            ocr_text = " ".join(extracted_lines).strip()            
        except Exception as e:
            logger.error(f"Erreur lors du processing OCR : {e}")
            ocr_text = ""
    
    # 3. LLM : Analyse de l'intention (Filtres intelligents)
    # On n'appelle le LLM que si on a assez de texte pour être utile
    if len(ocr_text) > 4:
        intent = llm.analyze_scan_intent(ocr_text)
    else:
        intent = {"domain": "unknown", "label": "unknown", "type": "image"}
    
    # ON RENVOIE TOUT : Le vecteur CLIP ET les filtres LLM
    return {
        "vector": vector,    # Pour LanceDB .search()
        "ocr_text": ocr_text,
        "filters": intent    # Pour LanceDB .where()
    }