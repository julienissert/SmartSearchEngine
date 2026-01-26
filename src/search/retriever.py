# src/search/retriever.py
import numpy as np
import pandas as pd
from src import config
from collections import Counter
from src.embeddings.text_embeddings import embed_text
from src.indexing.vector_store import init_tables 
from src.utils.logger import setup_logger

logger = setup_logger("SearchEngine", log_file="search.log")

class MultiDomainRetriever:
    def __init__(self):
        # init_tables() assure que la DB est connectée ET que la table est là
        self.table = init_tables()
        logger.info("Moteur de recherche hybride LanceDB prêt.")

    def search(self, image_vector, query_ocr="", k=5):
        logger.info(f"--- Requête Hybride (OCR: '{query_ocr.strip()}') ---")
        
        # 1. On récupère les candidats (Fusion Visuel + Textuel)
        all_candidates = self._get_hybrid_candidates(image_vector, query_ocr)
        
        if not all_candidates:
            return []

        # 2. On valide le label (Consensus entre les meilleurs résultats)
        verified_label = self._determine_verified_label(all_candidates, query_ocr)

        # 3. On construit la réponse finale (Agrégation par label ou Top K)
        results = self._build_final_response(all_candidates, verified_label, k)
        
        return results

    def _get_hybrid_candidates(self, image_vector, query_ocr):
        """Fusionne les résultats de recherche Image et Texte."""
        results_list = []

        # -- BRANCHE 1 : Recherche Visuelle (Image) --
        if image_vector is not None:
            v_img = np.array(image_vector).astype('float32')
            norm = np.linalg.norm(v_img)
            search_vec = (v_img / norm if norm > 0 else v_img).tolist()
            
            # Recherche Top 50 par similarité cosinus
            res_vis = self.table.search(search_vec).limit(50).to_pandas()
            if not res_vis.empty:
                res_vis['origin'] = 'visual'
                results_list.append(res_vis)

        # -- BRANCHE 2 : Recherche Textuelle (OCR / Query) --
        if query_ocr.strip():
            text_vec = embed_text(query_ocr)
            v_txt = np.array(text_vec).astype('float32')
            norm_t = np.linalg.norm(v_txt)
            search_vec_t = (v_txt / norm_t if norm_t > 0 else v_txt).tolist()

            res_txt = self.table.search(search_vec_t).limit(50).to_pandas()
            if not res_txt.empty:
                res_txt['origin'] = 'textual'
                res_txt['_distance'] = res_txt['_distance'] * 0.9 
                results_list.append(res_txt)

        if not results_list:
            return []

        combined = pd.concat(results_list).sort_values('_distance')
        combined = combined.drop_duplicates(subset=['source'])

        candidates = []
        for _, row in combined.iterrows():
            candidates.append({
                "score": row.get("_distance", 1.0),
                "domain": row["domain"],
                "label": row["label"].lower(),
                "type": row["type"],
                "source": row["source"],
                "content": row["content"],
                "snippet": row["snippet"]
            })
        return candidates

    def _determine_verified_label(self, candidates, query_ocr):
        """Garde ta logique de consensus Élite."""
        if query_ocr.strip():
            ocr_text = query_ocr.lower()
            potential_labels = {c["label"] for c in candidates[:15] if c["label"] != "unknown"}
            for lbl in potential_labels:
                if lbl in ocr_text: return lbl

        # Validation par vote majoritaire (Consensus)
        top_voters = [c["label"] for c in candidates[:15] if c["label"] != "unknown"]
        if top_voters:
            main_label, count = Counter(top_voters).most_common(1)[0]
            if count >= 3: return main_label
        return None

    def _build_final_response(self, candidates, target_label, k):
        """Finalise le pack de résultats."""
        if target_label:
            res_df = self.table.search().where(f"label = '{target_label}'").limit(k).to_pandas()
            source_items = res_df.to_dict('records')
        else:
            source_items = candidates[:k]

        final_results = []
        for item in source_items:
            final_results.append({
                "domain": item["domain"],
                "label": item["label"],
                "source": item["source"],
                "type": item["type"],
                "snippet": item.get("snippet", ""),
                "score": item.get("_distance", 0.0)
            })
        return final_results

retriever = MultiDomainRetriever()