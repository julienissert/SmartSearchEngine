# ingestion/csv_ingestion.py
import pandas as pd

def load_csv(path):
    """Charge un CSV et tente de trouver un label dans les colonnes."""
    try:
        df = pd.read_csv(path)
        # Nettoyage des noms de colonnes (enlève les espaces autour)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"Erreur CSV {path}: {e}")
        return []

    docs = []
    
    # LISTE MAGIQUE : Les colonnes qui contiennent souvent le "vrai nom"
    candidate_label_cols = ["Item", "Product", "Name", "Title", "Label", "Food", "Dish"]

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        
        # On cherche si une des colonnes candidates existe dans cette ligne
        suggested_label = None
        for col in candidate_label_cols:
            if col in row_dict and row_dict[col]:
                # On nettoie la valeur (ex: "Hamburger " -> "hamburger")
                val = str(row_dict[col]).strip()
                if len(val) > 2:
                    suggested_label = val.lower()
                    break # On a trouvé, on arrête
        
        doc = {
            "source": path,
            "type": "csv",
            "content": row_dict,
            # On passe ce "tuyau" au main.py
            "suggested_label": suggested_label 
        }
        docs.append(doc)

    return docs