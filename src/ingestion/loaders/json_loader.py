# src/ingestion/loaders/json_loader.py
import json
from src.ingestion.loaders.base_loader import BaseLoader

class JSONLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".json"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        """
        Approche généraliste : 
        - Si c'est une liste [], chaque élément est un document.
        - Si c'est un objet {}, le fichier entier est UN document.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return []

        records = data if isinstance(data, list) else [data]
        
        docs = []
        for record in records:
            if isinstance(record, dict):
                docs.append({
                    "source": path,
                    "type": "json",
                    "content": record, 
                    "suggested_label": None 
                })
            
        return docs