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
        except Exception as e:
            return []

        docs = []
        candidate_label_cols = ["Item", "Product", "Name", "Title", "Label"]

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            suggested_label = None
            for col in candidate_label_cols:
                if col in row_dict and row_dict[col]:
                    val = str(row_dict[col]).strip()
                    if len(val) > 2:
                        suggested_label = val.lower()
                        break 
            
            docs.append({
                "source": path,
                "type": "csv",
                "content": row_dict,
                "suggested_label": suggested_label 
            })
        return docs