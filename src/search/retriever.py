# src/search/retriever.py
import faiss
import os
import numpy as np
import config

class MultiDomainRetriever:
    def __init__(self):
        self.indices = {}
        self.load_indices()

    def load_indices(self):
        if not os.path.exists(config.FAISS_INDEX_DIR):
            return
            
        for file in os.listdir(config.FAISS_INDEX_DIR):
            if file.endswith(".index"):
                domain = file.replace(".index", "")
                path = os.path.join(config.FAISS_INDEX_DIR, file)
                self.indices[domain] = faiss.read_index(path)
                print(f"Index chargé : {domain}")

    def search(self, vector, k=5):
        results = []
        for domain, index in self.indices.items():
            distances, ids = index.search(np.array([vector]).astype("float32"), k)
            
            for dist, doc_id in zip(distances[0], ids[0]):
                if doc_id >= 0:
                    results.append({
                        "id": int(doc_id),
                        "score": float(dist),
                        "domain": domain
                    })
        
        return sorted(results, key=lambda x: x["score"])[:k]

retriever = MultiDomainRetriever()