# src/search/retriever.py
import faiss
import os
import numpy as np
import config
from embeddings.text_embeddings import embed_text  

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

    def search(self, image_vector, query_ocr="", k=5):
        all_matches = []
        
        for domain, index in self.indices.items():
            distances, ids = index.search(np.array([image_vector]).astype("float32"), k)
            
            for dist, doc_id in zip(distances[0], ids[0]):
                if doc_id >= 0:
                    all_matches.append({
                        "id": int(doc_id),
                        "score": float(dist),
                        "domain": domain,
                        "origin": "visual"
                    })

        if query_ocr.strip():
            text_vector = embed_text(query_ocr)
            
            for domain, index in self.indices.items():
                t_distances, t_ids = index.search(np.array([text_vector]).astype("float32"), k)
                
                for dist, doc_id in zip(t_distances[0], t_ids[0]):
                    if doc_id >= 0:
                        all_matches.append({
                            "id": int(doc_id),
                            "score": float(dist) * 0.9,
                            "domain": domain,
                            "origin": "textual"
                        })
        
        unique_results = {}
        for match in all_matches:
            key = (match["domain"], match["id"])
            if key not in unique_results or match["score"] < unique_results[key]["score"]:
                unique_results[key] = match
        
        final_sorted = sorted(unique_results.values(), key=lambda x: x["score"])
        return final_sorted[:k]

retriever = MultiDomainRetriever()