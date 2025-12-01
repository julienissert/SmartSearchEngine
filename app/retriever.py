# app/retriever.py
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import faiss
import numpy as np
import json

DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "meta.json")


class FaissRetriever:
    def __init__(self, dim=512):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.dim = dim

        # Load index if exists
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.index = faiss.IndexFlatL2(dim)

        # Load metadata
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.next_id = len(self.metadata)

    # ------------------------------
    # MAIN unified method
    # ------------------------------
    def add_vector(self, vector, metadata):
        """
        vector: 1D embedding (list or np array)
        metadata: dict
        """
        vector = np.array(vector, dtype="float32").reshape(1, -1)

        # Normalization for L2 index
        faiss.normalize_L2(vector)

        # Add vector
        self.index.add(vector)

        # Store metadata
        item_id = str(self.next_id)
        self.metadata[item_id] = metadata
        self.next_id += 1

        # Save
        self._save()

        return item_id

    # ------------------------------
    def search(self, query_vector, k=5):
        query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)

        D, I = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue

            item_id = str(idx)
            results.append({
                "distance": float(dist),
                "metadata": self.metadata[item_id]
            })

        return results

    # ------------------------------
    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)


# Singleton instance
retriever = FaissRetriever(dim=512)