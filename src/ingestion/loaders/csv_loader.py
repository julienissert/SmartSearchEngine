# src/ingestion/loaders/csv_loader.py
import pandas as pd
from src.ingestion.loaders.base_loader import BaseLoader

class CSVLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".csv"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        """Charge un CSV de manière ultra-rapide."""
        try:
            df = pd.read_csv(path, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
        except Exception:
            return []

        records = df.to_dict(orient='records')
        
        docs = []
        for record in records:
            docs.append({
                "source": path,
                "type": "csv",
                "content": record,
                "suggested_label": None 
            })
            
        return docs