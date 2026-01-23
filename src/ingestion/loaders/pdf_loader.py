# src/ingestion/loaders/pdf_loader.py
import fitz  
from src.ingestion.loaders.base_loader import BaseLoader

class PDFLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".pdf"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        text = ""
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            
            return [{
                "source": path,
                "type": "pdf",
                "content": text
            }]
        except Exception:
            return []