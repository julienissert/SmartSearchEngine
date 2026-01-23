# ingestion/loaders/txt_loader.py
import os
from src.utils.preprocessing import clean_text
from src.ingestion.loaders.base_loader import BaseLoader

class TXTLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".txt"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        docs = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            is_list = False
            if len(lines) > 2:
                sample = lines[:20]
                avg_len = sum(len(l.split()) for l in sample) / len(sample)
                if avg_len < 15:
                    is_list = True

            if is_list:
                for line in lines:
                    cleaned = clean_text(line)
                    if len(cleaned) > 2:
                        docs.append({
                            "source": path,
                            "type": "txt",
                            "content": cleaned,
                            "suggested_label": cleaned 
                        })
            else:
                cleaned = clean_text("".join(lines))
                docs.append({
                    "source": path,
                    "type": "txt",
                    "content": cleaned
                })
        except Exception:
            pass
        return docs