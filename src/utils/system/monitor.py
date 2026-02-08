# src/utils/system/monitor.py
import os
import psutil
import torch
import time
from src.utils.system import settings
from src.utils.logger import setup_logger

logger = setup_logger("Monitor")

class SystemMonitor:
    def __init__(self):
        self.total_ram = psutil.virtual_memory().total
        self.cpu_count = os.cpu_count() or 1

    def get_max_workers(self):
        # On réserve 12 Go pour le système et LanceDB
        available_ram = max(0, self.total_ram - (10 * 1024 * 1024 * 1024))
        ram_limit = int(available_ram / (2500 * 1024 * 1024)) 
        cpu_limit = max(1, self.cpu_count - 2)
        return max(1, min(cpu_limit, ram_limit, 8))

    def get_batch_size(self):
        if settings.DEVICE == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 20: return 512  
            if vram_gb >= 10: return 256   
            return 128                     
        return 64 # CPU

    def throttle(self):
        """Vérifie si le système sature et ordonne une pause si nécessaire."""
        ram_usage = psutil.virtual_memory().percent
        if ram_usage > 92:
            logger.warning(f"ALERTE RAM : {ram_usage}%. Pause de sécurité...")
            time.sleep(5)
            return True
        return False
    
    def get_cleanup_modulo(self):
        """
        Calcule dynamiquement la fréquence de nettoyage.
        Plus on a de VRAM, moins on nettoie souvent pour privilégier la vitesse.
        """
        if settings.DEVICE == "cuda":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb >= 20: return 20 
                if vram_gb >= 10: return 10  
                if vram_gb >= 6:  return 5   
                return 3                    
            except:
                return 5
        return 4

monitor = SystemMonitor()