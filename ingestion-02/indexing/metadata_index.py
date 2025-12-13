# indexing/metadata_index.py

# Liste globale qui servira de base de données en mémoire
metadata_store = []

def store_metadata(entry):
    """
    Stocke les métadonnées d'un document et lui assigne un ID unique.

    """
    entry["id"] = len(metadata_store)
    
    metadata_store.append(entry)

def search_by_label(label):
    """
    Récupère tous les documents ayant un label spécifique.
    """
    return [e for e in metadata_store if e.get("label") == label]

def get_all_metadata():
    """
    Renvoie toute la base de données.
    """
    return metadata_store

def get_metadata_by_id(doc_id):
    """
    Récupère un document spécifique par son ID (utile après une recherche FAISS).
    """
    try:
        return metadata_store[doc_id]
    except IndexError:
        return None