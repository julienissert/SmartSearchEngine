# src/search/processor.py
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from embeddings.image_embeddings import embed_image
from utils.logger import setup_logger

logger = setup_logger("Processor")

# --- Initialisation de PaddleOCR (Singleton) ---
# use_angle_cls=True : permet de détecter le texte même si l'image est tournée
# lang='fr' : charge le modèle français (qui gère très bien l'anglais aussi)
try:
    logger.info("Chargement du modèle PaddleOCR...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='fr', show_log=False)
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
                    
                    # Filtre optionnel : on peut ignorer le texte avec une confiance trop faible
                    if confidence > 0.5: 
                        extracted_lines.append(text_detected)
            
            ocr_text = " ".join(extracted_lines).strip()
            
        except Exception as e:
            logger.error(f"Erreur lors du processing OCR : {e}")
            ocr_text = ""
    else:
        logger.warning("Moteur OCR non disponible, analyse textuelle ignorée.")

    return vector, ocr_text