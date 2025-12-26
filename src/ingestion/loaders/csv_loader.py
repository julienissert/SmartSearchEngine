# ingestion/loaders/csv_loader.py
import pandas as pd
from ingestion.loaders.base_loader import BaseLoader

class CSVLoader(BaseLoader):
    def get_supported_extensions(self):
        return [".csv"]

    def can_handle(self, extension: str) -> bool:
        return extension.lower() in self.get_supported_extensions()

    def load(self, path: str, valid_labels=None) -> list:
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
        except Exception:
            return []

        docs = []
        for idx, row in df.iterrows():
            docs.append({
                "source": path,
                "type": "csv",
                "content": row.to_dict(),
                "suggested_label": None 
            })
        return docs