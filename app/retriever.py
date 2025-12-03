# app/retriever.py
import faiss
import numpy as np
import os
import json
from typing import List, Dict

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")

class FaissRetriever:
    def __init__(self, dim=512):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors
        self.metadatas = []  # list of dict
        # load if exists
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "r", encoding="utf8") as f:
                    self.metadatas = json.load(f)
            except Exception as e:
                print("Failed to load index:", e)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """
        embeddings: np.ndarray shape (N, dim) float32 (should be normalized)
        metadatas: list of dict len N
        """
        assert embeddings.shape[0] == len(metadatas)
        # ensure normalized
        # faiss.normalize_L2(embeddings)  # assume already normalized
        self.index.add(embeddings)
        self.metadatas.extend(metadatas)
        self._save()

    def search(self, query_emb: np.ndarray, k=5):
        q = query_emb.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            meta = self.metadatas[idx]
            results.append({"score": float(score), "meta": meta})
        return results

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)

# singleton
retriever = FaissRetriever(dim=512)