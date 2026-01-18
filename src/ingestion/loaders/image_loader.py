# src/ingestion/loaders/image_loader.py
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from ingestion.loaders.base_loader import BaseLoader

# Instance globale pour le processus en cours (évite de recharger le modèle à chaque fichier)
_ocr_engine = None

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        # Ajout de show_log=False pour désactiver les logs DEBUG de Paddle
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang='fr', show_log=False)
    return _ocr_engine

class ImageLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        try:
            # 1. Chargement Image
            img = Image.open(path).convert("RGB")
            
            # 2. Extraction OCR via Paddle
            engine = get_ocr_engine()
            img_array = np.array(img) # Paddle veut du Numpy, pas du PIL
            result = engine.ocr(img_array)
            
            ocr_text = ""
            if result and result[0]:
                # On concatène tout le texte trouvé avec une confiance > 60%
                texts = [line[1][0] for line in result[0] if line[1][1] > 0.6]
                ocr_text = " ".join(texts)

            return [{
                "source": path,
                "type": "image",
                "image": img,  # L'objet PIL est gardé pour la vectorisation CLIP plus tard
                "content": ocr_text
            }]
        except Exception as e:
            # On retourne une liste vide en cas d'erreur pour ne pas bloquer le workflow
            return []