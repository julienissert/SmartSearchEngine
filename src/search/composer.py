# src/search/composer.py
import config
from indexing.metadata_index import get_metadata_by_id

class ResultComposer:
    def build_response(self, matches, ocr_text):
        final_results = []
        
        for m in matches:
            domain = m["domain"]
            doc_id = m["id"] 
            
            # Interrogation directe de SQLite via l'ID
            data = get_metadata_by_id(doc_id, domain)
            
            if not data:
                continue
                
            res = {
                "domain": domain,
                "label": data.get("label"),
                "source": data.get("source"),
                "match_score": m["score"],
                "origin": m.get("origin", "unknown") 
            }

            # LOGIQUE D'ENRICHISSEMENT
            if data.get("raw_data"):
                res["details"] = data.get("raw_data")
            elif data.get("snippet"):
                res["details"] = data.get("snippet")
            
            final_results.append(res)
        
        return {
            "query_ocr": ocr_text,
            "top_results": final_results
        }

composer = ResultComposer()