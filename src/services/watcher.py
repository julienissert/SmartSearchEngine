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
        """Ignore les dossiers et les fichiers cachés (commençant par .)."""
        if not event.is_directory:
            filename = event.src_path.replace("\\", "/").split("/")[-1]
            if filename.startswith("."):
                return
            
            self.pending_event = True
            self.last_trigger_time = time.time()

    def on_created(self, event):
        self.process_event(event)

    def on_moved(self, event):
        self.process_event(event)
        
    def on_modified(self, event):
        self.process_event(event)

    def run_ingestion(self):
        """
        Mode 'r' (reset) si la base est vide, sinon 'c' (compléter).
        """
        mode = "c"
        # Vérification si la base de métadonnées existe et n'est pas vide
        db_path = config.LANCEDB_URI
        table_folder = db_path / f"{config.TABLE_NAME}.lance"        
        
        if not db_path.exists() or not table_folder.exists():
            mode = "r"
            logger.info("Base vide détectée -> Mode Réinitialisation (r)")
        
        logger.info(f"Déclenchement automatique de l'ingestion (Mode: {mode})")
        
        try:
            subprocess.run(
                [sys.executable, "-m", "src.main", "ingest", "-m", mode],
                check=True,
                cwd=str(config.BASE_DIR)
            )
            logger.info("Ingestion automatique terminée avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Échec de l'ingestion automatique (Code de sortie: {e.returncode}).")
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du processus d'ingestion : {e}")

def start_watching():
    """Point d'entrée principal du service de surveillance."""
    if not config.DATASET_DIR.exists():
        logger.error(f"Dossier à surveiller introuvable : {config.DATASET_DIR}")
        return

    handler = DatasetHandler(debounce_seconds=10)
    observer = Observer()
    observer.schedule(handler, str(config.DATASET_DIR), recursive=True)
    observer.start()
    
    logger.info(f"SmartSearch Watcher actif sur : {config.DATASET_DIR}")

    try:
        while True:
            if handler.pending_event:
                if (time.time() - handler.last_trigger_time) > handler.debounce_seconds:
                    handler.run_ingestion()
                    handler.pending_event = False
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Arrêt du Watcher...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    start_watching()