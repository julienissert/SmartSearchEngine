# src/search/scorer.py
from src.utils.preprocessing import compute_text_match_ratio
from src.utils.logger import setup_logger

logger = setup_logger("TrustScorer")

class TrustScorer:
    """
    Moteur de décision qui fusionne les preuves visuelles (CLIP) 
    et textuelles (OCR) pour garantir la précision sur 30 Go.
    """

    @staticmethod
    def calculate_score(result_row: dict, ocr_query: str, intent_filters: dict):
        """
        Formule  : Score = (Visuel * 0.5) + (Textuel * 0.4) + (Intention * 0.1)
        """
        # 1. PILIER VISUEL (50%) - Issu de la distance CLIP
        distance = result_row.get("_distance", 1.0)
        s_vis = max(0, 1 - distance)

        # 2. PILIER TEXTUEL (40%) - Utilise l'utilitaire de preprocessing
        # On compare l'OCR au label et au snippet fusionnés
        target_content = f"{result_row.get('label', '')} {result_row.get('snippet', '')}"
        s_txt = compute_text_match_ratio(ocr_query, target_content)

        # 3. PILIER INTENTION (10%) - Cohérence de domaine
        s_int = 1.0 if result_row.get("domain") == intent_filters.get("domain") else 0.0

        # Fusion finale
        final_score = (s_vis * 0.5) + (s_txt * 0.4) + (s_int * 0.1)
        
        return {
            "confidence": round(final_score * 100, 2),
            "details": {"visual": s_vis, "textual": s_txt, "intent": s_int}
        }
    