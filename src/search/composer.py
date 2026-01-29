# src/search/composer.py
import json
from src.utils.logger import setup_logger

logger = setup_logger("Composer")

class ResultComposer:
    def build_response(self, matches, ocr_text):
        """
        Prend les résultats du Retriever et les formate en faits bruts.
        Format : Domaine, Label, Informations Enrichies, Score.
        """
        final_results = []
        
        for m in matches:
            enriched_info = {}
            if m.get("extra"):
                try:
                    enriched_info = json.loads(m["extra"]) if isinstance(m["extra"], str) else m["extra"]
                except Exception:
                    enriched_info = {}

            res = {
                "domain": m.get("domain", "unknown"),
                "label": m.get("label", "unknown"),
                "confidence_score": m.get("confidence_score", 0.0),
                "metadata_enriched": enriched_info,
                "source_file": m.get("source", "unknown"),
                "match_details": m.get("confidence_details", {})
            }
            
            final_results.append(res)
        
        return {
            "status": "success" if final_results else "no_results",
            "query_ocr": ocr_text,
            "count": len(final_results),
            "results": final_results
        }

composer = ResultComposer()