# src/search/composer.py
import json

class ResultComposer:
    def build_response(self, matches, ocr_text):
        """
        Prend les résultats bruts de LanceDB et les formate proprement.
        Plus besoin de requêtes SQLite, tout est déjà dans 'matches'.
        """
        final_results = []
        
        for m in matches:
            res = {
                "domain": m["domain"],
                "label": m["label"],
                "source": m["source"],
                "type": m.get("type", "unknown"),
                "match_score": round(m["score"], 4),
                "origin": m.get("origin", "hybrid_search")
            }

            # LOGIQUE D'ENRICHISSEMENT (Snippet ou extra)
            if m.get("snippet"):
                res["details"] = m["snippet"]
            
            if m.get("extra"):
                try:
                    if isinstance(m["extra"], str):
                        res["metadata_extended"] = json.loads(m["extra"])
                    else:
                        res["metadata_extended"] = m["extra"]
                except:
                    pass
            
            final_results.append(res)
        
        return {
            "query_ocr": ocr_text,
            "total_count": len(final_results),
            "top_results": final_results
        }

composer = ResultComposer()