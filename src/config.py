# src/config.py
import os
import psutil 
from src.utils.system.settings import *
from src.utils.system.monitor import monitor

# --- SYSTEM RESOURCES ---
# On exporte les valeurs calculées dynamiquement
MAX_WORKERS = monitor.get_max_workers()
BATCH_SIZE = monitor.get_batch_size()
INGESTION_CHUNKSIZE = 10 
DYNAMIC_CACHE_SIZE = int(psutil.virtual_memory().total * 0.15 / -1024)
CLEANUP_MODULO = monitor.get_cleanup_modulo()

# --- LLM CONFIGURATION ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_GENERATE_URL = f"http://{OLLAMA_HOST}:11434/api/generate"

LLM_CONFIG = {
    "model": os.getenv("LLM_MODEL_NAME", "mistral"), 
    "temperature": 0.0,
    "num_ctx": 4096,
    "timeout": 300 
}

# --- AJOUT IMPORTANT (Correction de l'erreur) ---
# On crée un alias pour que environment.py puisse lire config.LLM_MODEL
LLM_MODEL = LLM_CONFIG["model"]

# --- BUSINESS LOGIC ---
TARGET_DOMAINS = os.getenv("TARGET_DOMAINS", "food,medical,cars").split(",")