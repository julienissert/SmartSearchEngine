# ingestion/loaders/pdf_loader.py
import PyPDF2
from ingestion.loaders.base_loader import BaseLoader

class PDFLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".pdf"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        text = ""
        try:
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return [{
                "source": path,
                "type": "pdf",
                "content": text
            }]
        except Exception:
            return []