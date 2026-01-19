# src/utils/watcher.py
import time
import subprocess
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src import config
from src.utils.logger import setup_logger

logger = setup_logger("Watcher")

class DatasetHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds=7):
        self.debounce_seconds = debounce_seconds
        self.last_trigger_time = 0
        self.pending_event = False

    def process_event(self, event):
        if not event.is_directory and not event.src_path.split("/")[-1].startswith("."):
            self.pending_event = True
            self.last_trigger_time = time.time()

    def on_created(self, event): self.process_event(event)
    def on_moved(self, event): self.process_event(event)

    def run_ingestion(self):
        # Choix automatique du mode
        mode = "c"
        if not config.METADATA_DIR.exists() or not any(config.METADATA_DIR.iterdir()):
            mode = "r"
            logger.info("Base vide détectée -> Mode Réinitialisation")
        
        logger.info(f"🚀 Déclenchement automatique (Mode: {mode})")
        try:
            # On utilise -m src.main pour rester dans le contexte package
            subprocess.run(
                [sys.executable, "-m", "src.main", "ingest", "-m", mode],
                check=True, cwd=str(config.BASE_DIR.parent)
            )
        except Exception as e:
            logger.error(f"Erreur Watcher : {e}")

def start_watching():
    handler = DatasetHandler()
    observer = Observer()
    observer.schedule(handler, str(config.DATASET_DIR), recursive=True)
    observer.start()
    logger.info(f"👀 Watcher actif sur : {config.DATASET_DIR}")
    try:
        while True:
            if handler.pending_event and (time.time() - handler.last_trigger_time) > handler.debounce_seconds:
                handler.run_ingestion()
                handler.pending_event = False
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()