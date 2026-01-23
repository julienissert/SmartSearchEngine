# src/config.py
import os
import psutil
from pathlib import Path
from dotenv import load_dotenv

# --- CRITIQUE : SÉCURITÉ ANTI-DEADLOCK ---
# Doit être défini AVANT l'import de torch/numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch  # Import torch seulement maintenant

# --- INITIALISATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def get_optimal_device():
    forced = os.getenv("DEVICE_OVERRIDE")
    if forced: return forced
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.zeros(1).to("cuda") 
            return "cuda"
        except Exception: return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_optimal_device()

class ResourceManager:
    def __init__(self):
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = os.cpu_count() or 1
        self.device = DEVICE

    def get_max_workers(self):
        # On réduit légèrement pour laisser respirer le système
        safe_ram = max(0, self.total_ram - (4 * 1024 * 1024 * 1024))
        ram_limit = int((safe_ram * 0.8) / (850 * 1024 * 1024))
        # Plafond à 10 workers max si on est sur GPU pour éviter la famine CPU
        # Sur CPU only, on peut monter plus haut
        cpu_limit = self.cpu_count - 2
        return max(1, min(cpu_limit, ram_limit, 20))

    def get_batch_size(self):
        if self.device == "cuda": return 256
        return 128 if self.cpu_count >= 16 else 64

    def get_chunksize(self):
        return max(5, min(50, 300 // self.get_max_workers()))

    def get_sql_cache_kb(self):
        budget = min(1024 * 1024 * 1024, int(self.total_ram * 0.05))
        return -int(budget / 1024)
        
    def get_hnsw_params(self):
        ram_gb = self.total_ram / (1024**3)
        M = 48 if ram_gb >= 64 else 32
        ef_c = 128
        ef_s = 64
        return M, ef_c, ef_s

res = ResourceManager()

# --- EXPORTS ---
MAX_WORKERS = res.get_max_workers()
BATCH_SIZE = res.get_batch_size()
INGESTION_CHUNKSIZE = res.get_chunksize()
DYNAMIC_CACHE_SIZE = res.get_sql_cache_kb()
FAISS_HNSW_M, FAISS_HNSW_EF_CONSTRUCTION, FAISS_HNSW_EF_SEARCH = res.get_hnsw_params()

OCR_FORCE_CPU = True
LABEL_BATCH_SIZE = 1000

if DEVICE == "cpu":
    torch.set_num_threads(1)

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM = 512
OCR_LANG = os.getenv("OCR_LANG", "latin")

DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_DIR = COMPUTED_DIR / "metadata"
METADATA_DB_PATH = COMPUTED_DIR / "metadata.db"

for path in [COMPUTED_DIR, FAISS_INDEX_DIR, METADATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

TARGET_DOMAINS = ["food", "medical"]
SEMANTIC_THRESHOLD = 0.65 
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
SEARCH_LARGE_K = 100
MAX_CLIP_CANDIDATES = 500
ENABLE_STATISTICAL_FALLBACK = True
FILE_READ_BUFFER_SIZE = 65536