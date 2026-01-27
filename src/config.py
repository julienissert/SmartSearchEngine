# src/config.py
import os
import psutil
from pathlib import Path
from dotenv import load_dotenv
import torch 

# --- CRITIQUE : SÉCURITÉ ANTI-DEADLOCK ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        safe_ram = max(0, self.total_ram - (8 * 1024 * 1024 * 1024))
        ram_limit = int((safe_ram * 0.75) / (1000 * 1024 * 1024))    
            
        if self.device != "cpu":
            cpu_limit = self.cpu_count 
        else:
            cpu_limit = int(self.cpu_count * 0.6) 
            
        return max(1, min(cpu_limit, ram_limit, 32))

    def get_torch_threads(self):
        if self.device != "cpu": return 1
        workers = self.get_max_workers()
        return max(1, self.cpu_count - workers)

    def get_batch_size(self):

        # --- LOGIQUE GPU (CUDA) ---
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 20: return 1024  
            if vram_gb >= 10: return 512   
            return 256                     

        # --- LOGIQUE CPU (Multi-niveaux) ---
        if self.cpu_count >= 30: return 512
        if self.cpu_count >= 16: return 256
        return 128 if self.cpu_count >= 8 else 64
    
    def get_chunksize(self):
        return 1

res = ResourceManager()

# --- EXPORTS ---
MAX_WORKERS = res.get_max_workers()
BATCH_SIZE = res.get_batch_size()
INGESTION_CHUNKSIZE = res.get_chunksize()

OCR_FORCE_CPU = True
LABEL_BATCH_SIZE = 1000

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

for path in [COMPUTED_DIR, LANCEDB_URI]:
    path.mkdir(parents=True, exist_ok=True)

# --- CHARGEMENT DYNAMIQUE DES DOMAINES ---
_env_domains = os.getenv("TARGET_DOMAINS")

if _env_domains:
    # Cas 1 : Variable d'environnement présente (Docker ou Prod)
    TARGET_DOMAINS = [d.strip() for d in _env_domains.split(",") if d.strip()]
else:
    # Cas 2 : Variable absente (Lancement local python src/main.py ...)
    # On définit des valeurs par défaut et on prévient l'utilisateur
    print("⚠️  Variable 'TARGET_DOMAINS' introuvable. Utilisation des domaines par défaut (Mode Local).")
    TARGET_DOMAINS = ["food", "medical"]

SEMANTIC_THRESHOLD = 0.65 
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50
SEARCH_LARGE_K = 100
MAX_CLIP_CANDIDATES = 500
ENABLE_STATISTICAL_FALLBACK = True
FILE_READ_BUFFER_SIZE = 65536