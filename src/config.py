# src/config.py
import os
import psutil
import torch
from pathlib import Path
from dotenv import load_dotenv

# --- INITIALISATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# --- LOGIQUE MATÉRIEL (DEVICE) ---
def get_optimal_device():
    """Détecte le meilleur accélérateur disponible (CUDA > MPS > CPU)."""
    forced = os.getenv("DEVICE_OVERRIDE")
    if forced: return forced
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            torch.zeros(1).to("cuda") 
            return "cuda"
        except Exception: return "cpu"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = get_optimal_device()

# --- GESTIONNAIRE DE RESSOURCES  ---
class ResourceManager:
    def __init__(self):
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = os.cpu_count() or 1
        self.device = DEVICE

    def _get_gpu_vram_gb(self):
        """Récupère la VRAM totale en Go si CUDA est actif."""
        if self.device == "cuda":
            try:
                props = torch.cuda.get_device_properties(0)
                return props.total_memory / (1024**3) 
            except: return 0
        return 0

    def get_max_workers(self):
        """
        Calcule les workers OCR selon la RAM Système (sécurité anti-swap).
        """
        # On réserve 4 Go de sécurité pour le système d'exploitation et CLIP
        safe_ram = max(0, self.total_ram - (4 * 1024 * 1024 * 1024))
        
        # Un worker PaddleOCR consomme ~850 Mo en charge
        ram_limit = int((safe_ram * 0.8) / (850 * 1024 * 1024))
        
        # On plafonne : au moins 1, max CPU-1, max 28 (rendement décroissant)
        return max(1, min(self.cpu_count - 1, ram_limit, 28))

    def get_batch_size(self):
        """
        Calcule la taille de lot CLIP (Batch Size).
        S'adapte à la VRAM (GPU) ou à la RAM (CPU) pour éviter le crash.
        """
        # 1. Base par puissance de calcul et VRAM
        if self.device == "cuda":
            vram = self._get_gpu_vram_gb()
            # Seuils adaptés aux cartes NVIDIA courantes
            if vram >= 20: base = 512    # A100, 3090, 4090
            elif vram >= 10: base = 256  # 3080, 2080 Ti, 1080 Ti
            elif vram >= 6: base = 128   # 3060, 2060, 1660
            elif vram >= 4: base = 64    # 1050 Ti, T600
            else: base = 32              # Vieilles cartes (<4Go)
            
        elif self.device == "mps":
            base = 256 # Mac Silicon (M1/M2/M3) gère bien la mémoire unifiée
        else:
            # Sur CPU, on évite les goulots d'étranglement mémoire
            base = 128 if self.cpu_count >= 16 else 64

        # 2. Sécurité RAM Système 
        ram_capacity = int((self.total_ram * 0.10) / (8 * 1024 * 1024))
        
        # 3. Alignement binaire (puissance de 2 pour perf matricielle optimale)
        optimized_size = min(base, ram_capacity)
        if optimized_size <= 8: return 8
        return 2**(optimized_size.bit_length() - 1)

    def get_chunksize(self):
        """Ajuste la distribution pour minimiser l'overhead IPC."""
        workers = self.get_max_workers()
        return max(5, min(50, 300 // workers))

    def get_sql_cache_kb(self):
        """Budget SQL : 5% de la RAM max, divisé par le nombre de workers."""
        budget_global = min(1024 * 1024 * 1024, int(self.total_ram * 0.05))
        return -int((budget_global / self.get_max_workers()) / 1024)
    
    def get_hnsw_params(self):
        """
        Calcule les paramètres HNSW selon la RAM et le CPU.
        M : Densité du graphe (impact RAM).
        ef_c : Précision construction (impact CPU).
        """
        ram_gb = self.total_ram / (1024**3)
        
        # 1. Ajustement de M (Densité)
        if ram_gb >= 64:   M = 48  # Serveur de calcul : précision max
        elif ram_gb >= 32: M = 32  # Machine pro standard
        elif ram_gb >= 16: M = 24  # Laptop correct
        else:              M = 16  # Configuration légère
        
        # 2. Ajustement de ef_construction (Temps d'ingestion vs Qualité)
        # Si on a beaucoup de coeurs, on peut se permettre une construction plus fine
        if self.cpu_count >= 16: ef_c = 128
        elif self.cpu_count >= 8: ef_c = 64
        else:                     ef_c = 40
        
        # 3. ef_search (Vitesse de réponse)
        # On reste sur une valeur équilibrée, ajustable par l'utilisateur
        ef_s = 64 if ram_gb >= 16 else 32
        
        return M, ef_c, ef_s


res = ResourceManager()

# --- EXPORTS DYNAMIQUES (Le coeur du système) ---
MAX_WORKERS = res.get_max_workers()
BATCH_SIZE = res.get_batch_size()
INGESTION_CHUNKSIZE = res.get_chunksize()
DYNAMIC_CACHE_SIZE = res.get_sql_cache_kb()
FAISS_HNSW_M, FAISS_HNSW_EF_CONSTRUCTION, FAISS_HNSW_EF_SEARCH = res.get_hnsw_params()

# Optimisation PyTorch CPU : Évite la "guerre des threads" (Oversubscription)
if DEVICE == "cpu":
    torch.set_num_threads(1)

# --- MODÈLES ---
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM = 512

# --- CHEMINS & AUTO-SETUP ---
DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_DIR = COMPUTED_DIR / "metadata"
METADATA_DB_PATH = COMPUTED_DIR / "metadata.db"

# Création automatique de l'arborescence (Robustesse Pro)
for path in [COMPUTED_DIR, FAISS_INDEX_DIR, METADATA_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- PARAMÈTRES MÉTIER (Recherche & Qualité) ---
TARGET_DOMAINS = ["food", "medical"]
SEMANTIC_THRESHOLD = 0.65 
CONSENSUS_THRESHOLD = 15
SEARCH_LARGE_K = 100
MAX_CLIP_CANDIDATES = 500
MAX_CONFIRMATION_IMAGES = 3
ENABLE_STATISTICAL_FALLBACK = True

# --- HYGIÈNE TECHNIQUE ---
FILE_READ_BUFFER_SIZE = 65536 
OCR_LANG = os.getenv("OCR_LANG", "latin")
LABEL_MIN_LENGTH = 3
LABEL_MAX_LENGTH = 50