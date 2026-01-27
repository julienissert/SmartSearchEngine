# src/config.py
import os
import psutil
from pathlib import Path
from dotenv import load_dotenv
import torch 

# --- PERFORMANCE CRITIQUE ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- INITIALISATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

def get_optimal_device():
    forced = os.getenv("DEVICE_OVERRIDE")
    if forced: return forced
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_optimal_device()

# --- OPTIMISATION FP16 (NVIDIA ONLY) ---
# Active le mode Mixed Precision si on est sur GPU Nvidia
USE_FP16 = (DEVICE == "cuda")

class ResourceManager:
    def __init__(self):
        self.total_ram = psutil.virtual_memory().total
        # On utilise cpu_count(logical=True) pour saturer l'Hyperthreading
        self.cpu_count = os.cpu_count() or 1
        self.device = DEVICE

    def get_max_workers(self):
        # STRATÉGIE : 90% de la RAM totale (Zone Rouge)
        target_ram_usage = self.total_ram * 0.90
        
        # Un worker OCR consomme ~700-800 Mo
        worker_ram_cost = 750 * 1024 * 1024 
        
        ram_limit = int(target_ram_usage / worker_ram_cost)

        # On utilise TOUS les cœurs moins 2 pour le système/GPU
        cpu_limit = max(1, self.cpu_count - 2)
            
        return max(1, min(cpu_limit, ram_limit))

    def get_torch_threads(self):
        return 1 # Toujours 1 pour éviter les conflits avec les Process

    def get_batch_size(self):
        # --- LOGIQUE GPU EXTRÊME ---
        if self.device == "cuda":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Base pour FP32
                if vram_gb >= 24: base = 2048
                elif vram_gb >= 16: base = 1024
                elif vram_gb >= 10: base = 512
                elif vram_gb >= 6:  base = 256
                else: base = 128
                
                # Si FP16 est activé, on peut souvent doubler le batch
                if USE_FP16:
                    base *= 2
                return base
            except:
                return 128

        # --- LOGIQUE CPU ---
        if self.cpu_count >= 32: return 512
        return 256 if self.cpu_count >= 16 else 128
    
    def get_chunksize(self):
        # Gros chunks pour minimiser l'overhead IPC
        return 10
    
    def get_sql_cache_kb(self):
        # 15% de la RAM pour le cache SQL
        budget = int(self.total_ram * 0.15)
        return -int(budget / 1024)
    
    def get_hnsw_params(self):
        # Paramètres très rapides pour l'insertion
        return 32, 100, 64

res = ResourceManager()

# --- EXPORTS ---
MAX_WORKERS = res.get_max_workers()
BATCH_SIZE = res.get_batch_size()
INGESTION_CHUNKSIZE = res.get_chunksize()
DYNAMIC_CACHE_SIZE = res.get_sql_cache_kb()
FAISS_HNSW_M, FAISS_HNSW_EF_CONSTRUCTION, FAISS_HNSW_EF_SEARCH = res.get_hnsw_params()

OCR_FORCE_CPU = True
LABEL_BATCH_SIZE = 4000 # Massif pour le démarrage

if DEVICE == "cpu":
    torch.set_num_threads(res.get_torch_threads())

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM = 512
OCR_LANG = os.getenv("OCR_LANG", "latin")

DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
LANCEDB_URI = COMPUTED_DIR / "lancedb_store"
TABLE_NAME = "multimodal_catalog"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_DIR = COMPUTED_DIR / "metadata"
METADATA_DB_PATH = COMPUTED_DIR / "metadata.db"

for path in [COMPUTED_DIR, FAISS_INDEX_DIR, METADATA_DIR, LANCEDB_URI]:
    path.mkdir(parents=True, exist_ok=True)

_env_domains = os.getenv("TARGET_DOMAINS")
if _env_domains:
    TARGET_DOMAINS = [d.strip() for d in _env_domains.split(",") if d.strip()]
else:
    print("⚠️  Variable 'TARGET_DOMAINS' introuvable. Mode Local par défaut.")
    TARGET_DOMAINS = ["food", "medical"]

SEMANTIC_THRESHOLD = 0.65 
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
SEARCH_LARGE_K = 100
MAX_CLIP_CANDIDATES = 500
ENABLE_STATISTICAL_FALLBACK = True
FILE_READ_BUFFER_SIZE = 131072 # Buffer lecture doublé