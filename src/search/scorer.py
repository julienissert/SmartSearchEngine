# src/search/scorer.py
from src.utils.preprocessing import compute_text_match_ratio
from src.utils.logger import setup_logger

logger = setup_logger("TrustScorer")

class TrustScorer:
    """Moteur de décision par fusion vectorielle normalisée."""

    @staticmethod
    def calculate_score(result_row: dict, ocr_query: str, intent_filters: dict):
        # 1. Preuve Visuelle (Distance brute LanceDB)
        # Similarity = 1 - Distance
        s_vis = max(0.0, min(1.0, 1.0 - result_row.get("_distance", 1.0)))

        # 2. Preuve Textuelle (Ratio de Levenshtein)
        target = f"{result_row.get('label', '')} {result_row.get('snippet', '')}"
        s_txt = compute_text_match_ratio(ocr_query, target) if ocr_query else 0.0

        # 3. Preuve d'Intention (Domaine)
        s_int = 1.0 if result_row.get("domain") == intent_filters.get("domain") else 0.0

        # --- CALCUL DE LA CONFIANCE PROUVÉE ---
        # Définition des coefficients de crédibilité
        w_vis = 0.6  # Poids Visuel
        w_txt = 0.3 if ocr_query else 0.0  # Poids Textuel (0 si pas de recherche texte)
        w_int = 0.1  # Poids Intention

        total_weight = w_vis + w_txt + w_int
        final_score = ((s_vis * w_vis) + (s_txt * w_txt) + (s_int * w_int)) / total_weight
        
        return {
            "confidence": round(final_score * 100, 2),
            "details": {"visual": s_vis, "textual": s_txt, "intent": s_int}
        }