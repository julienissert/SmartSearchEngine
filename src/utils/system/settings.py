# src/utils/system/settings.py
import os
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- INITIALISATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(BASE_DIR / ".env")

# --- PERFORMANCE CRITIQUE (THREADS) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_optimal_device():
    forced = os.getenv("DEVICE_OVERRIDE")
    if forced: return forced
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = get_optimal_device()
USE_FP16 = (DEVICE == "cuda")

# --- MODÈLES & DIMENSIONS ---
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM = 512
OCR_LANG = os.getenv("OCR_LANG", "latin")
OCR_FORCE_CPU = True

# --- CHEMINS & DOSSIERS ---
DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
LANCEDB_URI = COMPUTED_DIR / "lancedb_store"
TABLE_NAME = "multimodal_catalog"
METADATA_DB_PATH = COMPUTED_DIR / "metadata.db"
SCHEMA_CACHE_PATH = COMPUTED_DIR / "schema_cache.json"

# Création automatique des dossiers
for path in [COMPUTED_DIR, LANCEDB_URI]:
    path.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION DOMAINES ---
_env_domains = os.getenv("TARGET_DOMAINS")
TARGET_DOMAINS = [d.strip() for d in _env_domains.split(",") if d.strip()] if _env_domains else ["food", "medical"]
TECHNICAL_FOLDERS = ["images", "img", "photos", "train", "test", "meta", "archive", "dataset", "raw", "v1", "v2", "raw-datasets"]

# --- CONSTANTES DE FILTRAGE & RECHERCHE ---
SEMANTIC_THRESHOLD = 0.65 
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
LABEL_BATCH_SIZE = 4000
SEARCH_LARGE_K = 100
MAX_CLIP_CANDIDATES = 500
ENABLE_STATISTICAL_FALLBACK = True
FILE_READ_BUFFER_SIZE = 65536

# --- CONFIGURATION OLLAMA ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

def get_llm_resources():
    settings = {"model": "phi3:latest", "num_ctx": 2048, "timeout": 120}
    if DEVICE == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 10:
                settings.update({"model": "llama3:8b", "num_ctx": 8192, "timeout": 45})
            elif vram_gb >= 6:
                settings.update({"model": "llama3:latest", "num_ctx": 4096, "timeout": 60})
        except: pass 
    return settings

_res_llm = get_llm_resources()
LLM_CONFIG = {
    "model": _res_llm["model"],
    "temperature": 0.1,     
    "num_ctx": _res_llm["num_ctx"],
    "timeout": _res_llm["timeout"]
}
LLM_MODEL = LLM_CONFIG["model"]