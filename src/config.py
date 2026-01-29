# src/config.py
import psutil 
from src.utils.system.settings import * 
from src.utils.system.monitor import monitor

# On exporte les valeurs calculées dynamiquement
MAX_WORKERS = monitor.get_max_workers()
BATCH_SIZE = monitor.get_batch_size()
INGESTION_CHUNKSIZE = 10 
DYNAMIC_CACHE_SIZE = int(psutil.virtual_memory().total * 0.15 / -1024)
CLEANUP_MODULO = monitor.get_cleanup_modulo()