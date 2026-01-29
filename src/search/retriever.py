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

    def search(self, processed_query: dict, k: int = 5):
        """
        Recherche chirurgicale : Utilise le vecteur CLIP + les filtres SQL du LLM.
        """
        vector = processed_query["vector"]
        filters = processed_query["filters"]
        ocr_text = processed_query["ocr_text"]

        # 1. CONSTRUCTION DU FILTRE SQL (Metadata Filtering)
        sql_filter = self._build_sql_filter(filters)

        try:
            # 2. EXÉCUTION DE LA RECHERCHE VECTORIELLE FILTRÉE
            query = self.table.search(vector)
            
            if sql_filter:
                logger.info(f"Application des filtres SQL : {sql_filter}")
                query = query.where(sql_filter)
            
            results_df = query.limit(k).to_pandas()

            # 3. FALLBACK (Sécurité)
            if results_df.empty and sql_filter:
                logger.warning("Filtres trop stricts, fallback sur recherche vectorielle pure.")
                results_df = self.table.search(vector).limit(k).to_pandas()

            # 4. SCORING ET FORMATAGE
            final_results = []
            for _, row in results_df.iterrows():
                res_dict = row.to_dict()
                
                scoring = TrustScorer.calculate_score(res_dict, ocr_text, filters)
                
                # On enrichit le dictionnaire avec les scores
                res_dict["confidence_score"] = scoring["confidence"]
                res_dict["confidence_details"] = scoring["details"]
                final_results.append(res_dict)

            # Tri par score de confiance décroissant
            return sorted(final_results, key=lambda x: x["confidence_score"], reverse=True)

        except Exception as e:
            logger.error(f"Erreur lors de la recherche : {e}")
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