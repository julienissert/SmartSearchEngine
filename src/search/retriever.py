# src/search/retriever.py
import faiss
import os
import numpy as np
import config
from collections import Counter
from embeddings.text_embeddings import embed_text
from indexing.metadata_index import get_metadata_by_id, load_metadata_from_disk
from utils.logger import setup_logger

logger = setup_logger("SearchEngine", log_file="search.log")

class MultiDomainRetriever:
    def __init__(self):
        self.indices = {}
        load_metadata_from_disk()
        self.load_indices()

    def load_indices(self):
        if not os.path.exists(config.FAISS_INDEX_DIR):
            logger.warning(f"Dossier d'index introuvable : {config.FAISS_INDEX_DIR}")
            return
            
        for file in os.listdir(config.FAISS_INDEX_DIR):
            if file.endswith(".index"):
                domain = file.replace(".index", "")
                path = os.path.join(config.FAISS_INDEX_DIR, file)
                self.indices[domain] = faiss.read_index(path)
                logger.info(f"Index vectoriel chargé avec succès : {domain}")

    def search(self, image_vector, query_ocr="", k=5):
        logger.info(f"--- Nouvelle requête de recherche reçue (OCR: '{query_ocr.strip() if query_ocr else 'None'}') ---")
        
        # 1. RÉCUPÉRATION HYBRIDE (Vecteurs Image + Texte OCR)
        all_candidates = self._get_hybrid_candidates(image_vector, query_ocr)
        if not all_candidates:
            logger.warning("Aucun candidat n'a été trouvé dans les index vectoriels.")
            return []

        # 2. VALIDATION DÉTERMINISTE DU LABEL
        verified_label = self._determine_verified_label(all_candidates, query_ocr)

        # 3. EXTRACTION EXHAUSTIVE ET COMPOSITION
        results = self._build_final_response(all_candidates, verified_label, k)
        
        # Log de synthèse pour audit de performance
        img_count = len([r for r in results if r["origin"] == "visual_confirmation"])
        doc_count = len([r for r in results if r["origin"] == "enriched_info"])
        logger.info(f"Résultat final pour '{verified_label}': {img_count} images, {doc_count} documents enrichis.")
        
        return results

    def _get_hybrid_candidates(self, image_vector, query_ocr):
        candidates = []
        search_tasks = {"visual": image_vector}
        
        if query_ocr.strip():
            search_tasks["textual"] = embed_text(query_ocr)

        for mode, vector in search_tasks.items():
            for domain, index in self.indices.items():
                large_k = getattr(config, 'SEARCH_LARGE_K', 100)
                distances, ids = index.search(np.array([vector]).astype("float32"), large_k)
                
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
        
        logger.debug(f"Récupération brute terminée : {len(candidates)} candidats identifiés.")
        return sorted(candidates, key=lambda x: x["score"])

    def _determine_verified_label(self, candidates, query_ocr):
        
        # Étape A : Validation par OCR (Preuve textuelle directe)
        if query_ocr.strip():
            ocr_text = query_ocr.lower()
            potential_labels = {c["label"] for c in candidates[:15] if c["label"] != "unknown"}
            for lbl in potential_labels:
                if lbl in ocr_text:
                    logger.info(f"DÉCISION [OCR] : Label '{lbl}' validé par correspondance textuelle directe.")
                    return lbl

        # Étape B : Validation par Consensus (Vote majoritaire des voisins les plus proches)
        threshold = getattr(config, 'CONSENSUS_THRESHOLD', 15)
        top_voters = [c["label"] for c in candidates[:threshold] if c["label"] != "unknown"]
        if top_voters:
            main_label, count = Counter(top_voters).most_common(1)[0]
            logger.info(f"DÉCISION [CONSENSUS] : Label '{main_label}' validé par majorité ({count}/{len(top_voters)} voisins).")
            return main_label

        # Étape C : Défaut (Premier résultat mathématique)
        fallback_label = candidates[0]["label"]
        logger.warning(f"DÉCISION [FALLBACK] : Aucun consensus fort. Utilisation du premier voisin : '{fallback_label}'.")
        return fallback_label

    def _build_final_response(self, candidates, target_label, k):
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
                c["origin"] = "enriched_info"
                enriched_data.append(c)

        return confirmation_images + enriched_data

retriever = MultiDomainRetriever()