#indexing/faiss_index.py
import os
import faiss
import numpy as np
import config

def get_index_path(domain):
    return os.path.join(config.FAISS_INDEX_DIR, f"{domain}.index")

def load_or_create_index(domain):
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    path = get_index_path(domain)

    if os.path.exists(path):
        return faiss.read_index(path)

    index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    faiss.write_index(index, path)
    return index

def add_to_index(domain, vector):
    index = load_or_create_index(domain)
    index.add(np.array([vector]).astype("float32"))
    faiss.write_index(index, get_index_path(domain))
