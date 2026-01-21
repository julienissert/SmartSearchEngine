# src/ingestion/loaders/tsv_loader.py
import pandas as pd
from ingestion.loaders.base_loader import BaseLoader

class TSVLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".tsv"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        """Charge un TSV (Tab-Separated Values) de manière performante."""
        try:
            df = pd.read_csv(path, sep='\t', low_memory=False)
            df.columns = [c.strip() for c in df.columns]
        except Exception:
            return []

        records = df.to_dict(orient='records')
        
        docs = []
        for record in records:
            docs.append({
                "source": path,
                "type": "tsv",
                "content": record,
                "suggested_label": None 
            })
            
        return docs