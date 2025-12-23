# src/ingestion/loaders/image_loader.py
from PIL import Image
import pytesseract
from ingestion.loaders.base_loader import BaseLoader

class ImageLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        try:
            img = Image.open(path).convert("RGB")
            ocr_text = pytesseract.image_to_string(img)
            return [{
                "source": path,
                "type": "image",
                "image": img, 
                "content": ocr_text
            }]
        except Exception:
            return []