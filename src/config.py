# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv  
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env") 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- MODÈLES ---
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")

# --- CHEMINS ---
DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_DIR = COMPUTED_DIR / "metadata"

# --- PARAMÈTRES ---
EMBEDDING_DIM = 512
TARGET_DOMAINS = ["food", "medical"]
SEARCH_LARGE_K = 100
CONSENSUS_THRESHOLD = 15
MAX_CONFIRMATION_IMAGES = 3
SEMANTIC_THRESHOLD = 0.65 
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
ENABLE_STATISTICAL_FALLBACK = True
MAX_CLIP_CANDIDATES = 500

# --- PERFORMANCE ---
BATCH_SIZE = 256 
INGESTION_CHUNKSIZE = 20 # Nombre de fichiers par worker
FILE_READ_BUFFER_SIZE = 65536 # Hashing 

OCR_LANG = os.getenv("OCR_LANG", "latin")
METADATA_DB_PATH = COMPUTED_DIR / "metadata.db"

# --- LOGIQUE MATÉRIEL (DEVICE) ---
def get_optimal_device():
    forced = os.getenv("DEVICE_OVERRIDE")
    if forced: return forced
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.zeros(1).to("cuda") 
            return "cuda"
        except Exception:
            return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_optimal_device()