# ingestion/csv_ingestion.py
import pandas as pd

def load_csv(path):
    """Charge un CSV et tente de trouver un label dans les colonnes."""
    try:
        df = pd.read_csv(path)
        # Nettoyage des noms de colonnes
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Erreur CSV {path}: {e}")
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
        
        doc = {
            "source": path,
            "type": "csv",
            "content": row_dict,
            "suggested_label": suggested_label 
        }
        docs.append(doc)

    return docs