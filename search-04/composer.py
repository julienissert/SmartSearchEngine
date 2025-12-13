# app/composer.py
from typing import List, Dict

def detect_domain_from_matches(matches: List[Dict]) -> str:
    """
    Each metadata should include 'domain' field.
    We pick the majority domain among matches.
    """
    domains = {}
    for m in matches:
        d = m["meta"].get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    if not domains:
        return "unknown"
    # return domain with max count
    return max(domains.items(), key=lambda x: x[1])[0]

def compose_result(image_labels, nutrition_matches):
    """
    Combine best image label and best nutrition match (if domain == 'food').
    nutrition_matches: list of {'score', 'meta': {id, name, calories, text, domain}}
    """
    top_label = image_labels[0] if image_labels else {"label": "unknown", "score": 0.0}
    # find nutrition match with same entity name if possible
    if nutrition_matches:
        best = nutrition_matches[0]["meta"]
        return {
            "detected_label": top_label["label"],
            "label_score": top_label["score"],
            "nutrition": {
                "id": best.get("id"),
                "name": best.get("name"),
                "calories_per_100g": best.get("calories_per_100g"),
                "raw_text": best.get("text")
            }
        }
    else:
        return {
            "detected_label": top_label["label"],
            "label_score": top_label["score"],
            "nutrition": None
        }