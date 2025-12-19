# src/search/composer.py
import json
import os 
import config

class ResultComposer:
    def __init__(self):
        self.metadata = []
        self.refresh()

    def refresh(self):
        if os.path.exists(config.METADATA_FILE):
            with open(config.METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def build_response(self, matches, ocr_text):
        final_results = []
        for m in matches:
            try:
                doc_id = m["id"]
                data = self.metadata[doc_id]
                
                if data.get("domain") == m["domain"]:
                    final_results.append({
                        "domain": m["domain"],
                        "label": data.get("label"),
                        "source": data.get("source"),
                        "snippet": data.get("snippet"),
                        "match_score": m["score"]
                    })
            except (IndexError, KeyError):
                continue
        
        return {
            "query_ocr": ocr_text,
            "top_results": final_results
        }

composer = ResultComposer()