import os
from utils.preprocessing import clean_text

def load_txt(path: str):
    docs = []
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    is_list = False
    if len(lines) > 2:
        sample = lines[:20]
        avg_len = sum(len(l.split()) for l in sample) / len(sample)
        if avg_len < 15:
            is_list = True

    if is_list:
        print(f" -> Liste ingérée : {os.path.basename(path)}")
        for line in lines:
            cleaned = clean_text(line)
            if len(cleaned) > 2:
                docs.append({
                    "source": path,
                    "type": "txt",
                    "content": cleaned,
                    "suggested_label": cleaned 
                })
    else:
        cleaned = clean_text("".join(lines))
        docs.append({
            "source": path,
            "type": "txt",
            "content": cleaned
        })
    
    return docs