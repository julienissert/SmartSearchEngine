# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv  
import torch
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env") 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

#config label_detector.py
# --- SEUILS D'INTELLIGENCE & DÉTECTION ---
# Confiance minimale pour CLIP (Couche 2)
SEMANTIC_THRESHOLD = 0.65 

# Paramètres statistiques (Couche 3)
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
ENABLE_STATISTICAL_FALLBACK = True # Toujours True par défaut selon ton choix

# Limites de performance
MAX_CLIP_CANDIDATES = 500

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" 
else:
    DEVICE = "cpu"
    
BATCH_SIZE = 32 # Nombre de documents traités simultanément CLIP
