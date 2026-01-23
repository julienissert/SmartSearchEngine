# src/search/retriever.py
import faiss
import os
import numpy as np
from src import config
from collections import Counter
from src.embeddings.text_embeddings import embed_text
from src.indexing.metadata_index import get_metadata_by_id, load_metadata_from_disk, get_metadata_by_label
from src.utils.logger import setup_logger

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
        logger.info(f"--- Nouvelle requête reçue (OCR: '{query_ocr.strip() if query_ocr else 'None'}') ---")
        
        all_candidates = self._get_hybrid_candidates(image_vector, query_ocr)
        if not all_candidates:
            return []

        # 1. On tente de valider un label (Consensus/OCR)
        verified_label = self._determine_verified_label(all_candidates, query_ocr)

        # 2. On construit la réponse (Mode Agrégé si label trouvé, sinon Mode FAISS standard)
        results = self._build_final_response(all_candidates, verified_label, k)
        
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
        return sorted(candidates, key=lambda x: x["score"])

    def _determine_verified_label(self, candidates, query_ocr):
        # Étape A : Validation par OCR 
        if query_ocr.strip():
            ocr_text = query_ocr.lower()
            potential_labels = {c["label"] for c in candidates[:15] if c["label"] != "unknown"}
            for lbl in potential_labels:
                if lbl in ocr_text:
                    logger.info(f"DÉCISION [OCR] : Label '{lbl}' validé.")
                    return lbl

        # Étape B : Validation par Consensus 
        threshold = getattr(config, 'CONSENSUS_THRESHOLD', 15)
        top_voters = [c["label"] for c in candidates[:threshold] if c["label"] != "unknown"]
        if top_voters:
            main_label, count = Counter(top_voters).most_common(1)[0]
            # Sécurité : au moins 3 voisins doivent être d'accord
            if count >= 3:
                logger.info(f"DÉCISION [CONSENSUS] : Label '{main_label}' validé ({count} votes).")
                return main_label
        
        logger.info("Aucun label n'a pu être validé de manière déterministe.")
        return None

    def _build_final_response(self, candidates, target_label, k):
        """
        LOGIQUE HYBRIDE ÉTAPE 5 :
        - Si label valide : SQL Aggregation 
        - Si pas de label : Fallback sur les meilleurs résultats FAISS
        """
        confirmation_images = []
        enriched_data = []
        max_imgs = getattr(config, 'MAX_CONFIRMATION_IMAGES', 3)

        if target_label:
            # --- MODE AGRÉGATION  ---
            source_list = get_metadata_by_label(target_label, limit=config.BATCH_SIZE)
        else:
            # --- MODE STANDARD ---
            source_list = []
            for c in candidates[:k]:
                meta = get_metadata_by_id(c["id"], c["domain"])
                if meta: source_list.append(meta)

        for meta in source_list:
            result_item = {
                "id": meta["local_id"],
                "domain": meta["domain"],
                "label": meta["label"],
                "type": meta["type"],
                "source": meta["source"],
                "snippet": meta.get("snippet", ""),
                "raw_data": meta.get("raw_data"),
                "score": 0.0, 
                "origin": ""
            }

            if meta["type"] == "image":
                if len(confirmation_images) < max_imgs:
                    result_item["origin"] = "visual_confirmation"
                    confirmation_images.append(result_item)
            else:
                result_item["origin"] = "enriched_info"
                enriched_data.append(result_item)

        return confirmation_images + enriched_data

retriever = MultiDomainRetriever()