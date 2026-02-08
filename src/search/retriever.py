# src/search/retriever.py
import pandas as pd
from src.indexing.vector_store import init_tables 
from src.search.scorer import TrustScorer
from src.utils.logger import setup_logger

logger = setup_logger("SearchEngine")

class MultiDomainRetriever:
    def __init__(self):
        # Initialisation unique de la table LanceDB
        self.table = init_tables()
        logger.info("Moteur de recherche hybride LanceDB prêt.")

    def search(self, processed_query: dict, k: int = 10):
        """Recherche Tri-Pass : Visuelle Pure, Fusionnée et Label."""
        fused_vec = processed_query["fused_vector"]
        pure_vec = processed_query["pure_visual_vector"]
        filters = processed_query["filters"]
        ocr_text = processed_query["ocr_text"]

        all_matches = []

        try:
            # PASS 1 : Recherche Visuelle Pure (100% Précision Image)
            # On cherche spécifiquement dans la colonne sans pollution textuelle
            res_pure = self.table.search(pure_vec, vector_column_name="visual_pure").limit(k).to_pandas()
            all_matches.append(res_pure)

            # PASS 2 : Recherche Fusionnée (Sémantique & Contexte)
            res_fused = self.table.search(fused_vec, vector_column_name="vector").limit(k).to_pandas()
            all_matches.append(res_fused)

            # PASS 3 : Recherche par Label (FTS / SQL LIKE)
            sql_filter = self._build_sql_filter(filters)
            if sql_filter:
                res_label = self.table.search(fused_vec).where(sql_filter).limit(k).to_pandas()
                all_matches.append(res_label)

            # FUSION ET DÉDOUBLONNAGE
            combined_df = pd.concat(all_matches).drop_duplicates(subset=['file_hash'])
            
            # SCORING FINAL
            final_results = []
            for _, row in combined_df.iterrows():
                res_dict = row.to_dict()
                scoring = TrustScorer.calculate_score(res_dict, ocr_text, filters)
                res_dict["confidence_score"] = scoring["confidence"]
                res_dict["confidence_details"] = scoring["details"]
                final_results.append(res_dict)

            return sorted(final_results, key=lambda x: x["confidence_score"], reverse=True)

        except Exception as e:
            logger.error(f"Erreur Tri-Pass : {e}")
            return []

    def _build_sql_filter(self, filters: dict) -> str:
        """Transforme l'intention du LLM en clause SQL."""
        clauses = []
        
        # Filtrage par domaine
        domain = filters.get("domain")
        if domain and domain != "unknown":
            clauses.append(f"domain = '{domain}'")
        
        # Filtrage par label 
        label = filters.get("label")
        if label and label != "unknown":
            label_esc = str(label).replace("'", "''")
            clauses.append(f"label LIKE '%{label_esc}%'")
            
        return " AND ".join(clauses)

retriever = MultiDomainRetriever()