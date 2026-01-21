# src/ingestion/dispatcher.py
import os
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.loaders.h5_loader import H5Loader
from ingestion.loaders.txt_loader import TXTLoader
from ingestion.loaders.tsv_loader import TSVLoader
from ingestion.loaders.json_loader import JSONLoader

# 1. Mapping direct Extension -> Classe 
LOADER_MAPPING = {
    '.csv': CSVLoader,
    '.tsv': TSVLoader,    
    '.pdf': PDFLoader,
    '.h5': H5Loader,
    '.json': JSONLoader,
    '.txt': TXTLoader,
    '.jpg': ImageLoader, '.jpeg': ImageLoader, '.png': ImageLoader, '.webp': ImageLoader
}

# 2. Cache d'instances 
LOADER_CACHE = {}

def get_supported_extensions():
    return list(LOADER_MAPPING.keys())

def dispatch_loader(path: str, valid_labels=None):
    ext = os.path.splitext(path)[1].lower()
    loader_class = LOADER_MAPPING.get(ext)
    
    if not loader_class:
        raise ValueError(f"Aucun loader disponible pour l'extension {ext}")

    if loader_class not in LOADER_CACHE:
        LOADER_CACHE[loader_class] = loader_class()
    
    return LOADER_CACHE[loader_class].load(path, valid_labels=valid_labels)