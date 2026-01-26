# src/ingestion/loaders/pdf_loader.py
import fitz  
from src.ingestion.loaders.base_loader import BaseLoader

class PDFLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".pdf"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        pages_content = []
        try:
            with fitz.open(path) as doc:
                for page in doc:

                    page_text = str(page.get_text("text"))
                    pages_content.append(page_text)
            
            full_text = "\n".join(pages_content)
            
            return [{
                "source": path,
                "type": "pdf",
                "content": full_text
            }]
        except Exception:
            return []