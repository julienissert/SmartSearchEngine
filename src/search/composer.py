# src/search/composer.py
import json
import os 
import config

class ResultComposer:
    def __init__(self):
        self.metadata_by_domain = {}
        self.refresh()

    def refresh(self):
        self.metadata_by_domain = {}
        if os.path.exists(config.METADATA_DIR):
            for file in os.listdir(config.METADATA_DIR):
                if file.endswith(".json"):
                    domain = file.replace(".json", "")
                    path = config.METADATA_DIR / file
                    with open(path, "r", encoding="utf-8") as f:
                        self.metadata_by_domain[domain] = json.load(f)

    def build_response(self, matches, ocr_text):
        final_results = []
        for m in matches:
            domain = m["domain"]
            doc_id = m["id"]
            try:
                domain_data = self.metadata_by_domain.get(domain, [])
                data = domain_data[doc_id]
                
                # On prépare la base du résultat
                res = {
                    "domain": domain,
                    "label": data.get("label"),
                    "source": data.get("source"),
                    "match_score": m["score"],
                    "origin": m.get("origin", "unknown") 
                }

                # LOGIQUE D'AFFICHAGE ENRICHIE
                # 1. Si on a des données CSV/H5 complètes, on les met dans 'details'
                if data.get("raw_data"):
                    res["details"] = data.get("raw_data")
                # 2. Sinon, si on a un snippet texte (PDF/Image), on le met dans 'details'
                elif data.get("snippet"):
                    res["details"] = data.get("snippet")
                # 3. Si rien du tout (médicament simple), 'details' n'apparaît pas
                
                final_results.append(res)

            except (IndexError, KeyError):
                continue
        
        return {
            "query_ocr": ocr_text,
            "top_results": final_results
        }

composer = ResultComposer()