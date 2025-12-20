# src/utils/logger.py
import logging
import os
import sys
import gzip
import shutil
from logging.handlers import RotatingFileHandler

class CompressedRotatingFileHandler(RotatingFileHandler):
    
    def doRollover(self):
        super().doRollover()
        
        old_log = self.baseFilename + ".1"
        if os.path.exists(old_log):
            with open(old_log, 'rb') as f_in:
                with gzip.open(old_log + ".gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(old_log)

def setup_logger(name, log_file="ingestion.log", level=logging.INFO):
    
    os.makedirs("logs", exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Handler Console 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 2. Handler Fichier avec Compression et Rotation
    # backupCount = 5 
    file_path = os.path.join("logs", log_file)
    file_handler = CompressedRotatingFileHandler(
        file_path, 
        maxBytes=5*1024*1024, 
        backupCount=5, 
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger