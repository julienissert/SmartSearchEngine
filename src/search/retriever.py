# src/search/retriever.py
import faiss
import os
import numpy as np
import config
from collections import Counter
from embeddings.text_embeddings import embed_text
from indexing.metadata_index import get_metadata_by_id, load_metadata_from_disk

class MultiDomainRetriever:
    def __init__(self):
        self.indices = {}
        load_metadata_from_disk()
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
        # 1. RÉCUPÉRATION HYBRIDE (Vecteurs Image + Texte OCR)
        all_candidates = self._get_hybrid_candidates(image_vector, query_ocr)
        if not all_candidates:
            return []

        # 2. VALIDATION DÉTERMINISTE DU LABEL (Pipeline de décision)
        verified_label = self._determine_verified_label(all_candidates, query_ocr)

        # 3. EXTRACTION EXHAUSTIVE DES DONNÉES ENRICHIES
        return self._build_final_response(all_candidates, verified_label, k)

    def _get_hybrid_candidates(self, image_vector, query_ocr):
        candidates = []
        search_tasks = {"visual": image_vector}
        
        if query_ocr.strip():
            search_tasks["textual"] = embed_text(query_ocr)

        for mode, vector in search_tasks.items():
            for domain, index in self.indices.items():
                # On cherche large (K=100) pour trouver le texte derrière les images
                distances, ids = index.search(
                    np.array([vector]).astype("float32"), 
                    getattr(config, 'SEARCH_LARGE_K', 100)
                )
                
                for dist, doc_id in zip(distances[0], ids[0]):
                    if doc_id >= 0:
                        meta = get_metadata_by_id(int(doc_id), domain)
                        if meta:
                            candidates.append({
                                "id": int(doc_id),
                                "score": float(dist) * (0.8 if mode == "textual" else 1.0),
                                "domain": domain,
                                "label": meta.get("label", "unknown").lower(),
                                "type": meta.get("type", "image")
                            })
        
        # Tri par pertinence mathématique globale
        return sorted(candidates, key=lambda x: x["score"])

    def _determine_verified_label(self, candidates, query_ocr):
        """Logique de décision : Priorité OCR > Consensus Visuel > Plus proche voisin."""
        
        # Étape A : Validation par OCR (Preuve textuelle directe)
        if query_ocr.strip():
            ocr_text = query_ocr.lower()
            # On vérifie si un label des top candidats est écrit sur l'image
            potential_labels = {c["label"] for c in candidates[:15] if c["label"] != "unknown"}
            for lbl in potential_labels:
                if lbl in ocr_text:
                    return lbl

        # Étape B : Validation par Consensus (Vote majoritaire des voisins)
        threshold = getattr(config, 'CONSENSUS_THRESHOLD', 15)
        top_voters = [c["label"] for c in candidates[:threshold] if c["label"] != "unknown"]
        if top_voters:
            return Counter(top_voters).most_common(1)[0][0]

        # Étape C : Défaut sur le premier résultat
        return candidates[0]["label"]

    def _build_final_response(self, candidates, target_label, k):
        """Filtre les résultats pour ne garder que le label validé et sépare images/données."""
        confirmation_images = []
        enriched_data = []
        seen_keys = set()
        
        max_imgs = getattr(config, 'MAX_CONFIRMATION_IMAGES', 3)

        for c in candidates:
            if c["label"] != target_label:
                continue

            unique_key = f"{c['domain']}_{c['id']}"
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)

            if c["type"] == "image":
                if len(confirmation_images) < max_imgs:
                    c["origin"] = "visual_confirmation"
                    confirmation_images.append(c)
            else:
                # ASPIRATION : On prend TOUTE l'information enrichie trouvée (CSV, PDF, TXT)
                c["origin"] = "enriched_info"
                enriched_data.append(c)

        # Fusion : on retourne les images de preuve suivies de toute la connaissance textuelle
        return confirmation_images + enriched_data

retriever = MultiDomainRetriever()