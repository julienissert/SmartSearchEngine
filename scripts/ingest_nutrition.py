# scripts/ingest_nutrition.py
"""
Usage:
python scripts/ingest_nutrition.py nutrition.csv

CSV example columns: name, description, calories_per_100g
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.append(ROOT)
import pandas as pd
import numpy as np
from app.retriever import retriever
from app.embed import embed_text


def main(path):
    df = pd.read_csv(path)
    items = []
    for _, row in df.iterrows():
        name = str(row.get("food", ""))
        desc = str(row.get("description", name))
        calories = row.get("Caloric Value", None)
        emb = embed_text(desc)
        meta = {"id": f"nutrition/{len(retriever.metadatas)+len(items)}", "domain": "food", "name": name, "Caloric Value": float(calories) if calories is not None else None, "text": desc}
        items.append((emb, meta))
    if items:
        embs = np.vstack([e for e, m in items]).astype("float32")
        metas = [m for e, m in items]
        retriever.add(embs, metas)
    print(f"Indexed {len(items)} nutrition items. Total indexed: {len(retriever.metadatas)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_nutrition.py nutrition.csv")
        sys.exit(1)
    main(sys.argv[1])