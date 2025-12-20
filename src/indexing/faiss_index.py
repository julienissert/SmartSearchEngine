# src/indexing/faiss_index.py
import os
import shutil  
import faiss
import numpy as np
import config

indexes = {}

def load_all_indexes():
    global indexes
    if not os.path.exists(config.FAISS_INDEX_DIR):    
        return
    
    for filename in os.listdir(config.FAISS_INDEX_DIR):
        if filename.endswith(".index"):
            domain = filename.replace(".index", "")
            path = os.path.join(config.FAISS_INDEX_DIR, filename)
            indexes[domain] = faiss.read_index(path)
            
def get_index_path(domain):
    return os.path.join(config.FAISS_INDEX_DIR, f"{domain}.index")

def reset_all_indexes():
    if os.path.exists(config.FAISS_INDEX_DIR):
        shutil.rmtree(config.FAISS_INDEX_DIR)
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    print(f"Index FAISS réinitialisés dans : {config.FAISS_INDEX_DIR}")

def load_or_create_index(domain):
    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    path = get_index_path(domain)

    if os.path.exists(path):
        return faiss.read_index(path)

    base_index = faiss.IndexFlatL2(config.EMBEDDING_DIM)
    index = faiss.IndexIDMap(base_index) 
    
    faiss.write_index(index, path)
    return index

def add_to_index(domain, vector, doc_id):
    index = load_or_create_index(domain)
    
    v = np.array([vector]).astype("float32")
    ids = np.array([doc_id]).astype("int64")
    
    index.add_with_ids(v, ids)
    
    faiss.write_index(index, get_index_path(domain))