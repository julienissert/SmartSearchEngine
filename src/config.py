# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv  

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env") 

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")

DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_DIR = COMPUTED_DIR / "metadata"
EMBEDDING_DIM = 512
TARGET_DOMAINS = ["food", "medical"]

# Paramètres de Recherche
SEARCH_LARGE_K = 100       # Profondeur d'exploration pour trouver le texte
CONSENSUS_THRESHOLD = 15   # Nombre de voisins pour voter le label
MAX_CONFIRMATION_IMAGES = 3 # Nombre d'images de preuve à garder