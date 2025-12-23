# src/ingestion/dispatcher.py
import os
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.loaders.h5_loader import H5Loader
from ingestion.loaders.txt_loader import TXTLoader

# Enregistrement manuel des plugins
LOADERS = [
    CSVLoader(),
    PDFLoader(),
    ImageLoader(),
    H5Loader(),
    TXTLoader()
]

def get_supported_extensions():
    """Source de vérité unique pour les extensions gérées."""
    extensions = []
    for loader in LOADERS:
        extensions.extend(loader.get_supported_extensions())
    return list(set(extensions))

def dispatch_loader(path: str, valid_labels=None):
    ext = os.path.splitext(path)[1].lower()
    for loader in LOADERS:
        if loader.can_handle(ext):
            return loader.load(path, valid_labels=valid_labels)
    raise ValueError(f"Aucun loader disponible pour l'extension {ext}")