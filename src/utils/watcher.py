import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Imports internes
from src.utils.logger import setup_logger
from src import config

logger = setup_logger("Watcher")

class DatasetHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds=5):
        self.debounce_seconds = debounce_seconds
        self.last_trigger_time = 0
        self.pending_event = False
        # Chemin absolu vers le script d'ingestion pour éviter les erreurs de contexte
        self.ingestion_script = config.BASE_DIR / "src" / "ingestion" / "main.py"

    def process_event(self, event):
        """Logique commune pour création et déplacement de fichiers."""
        if not event.is_directory:
            # On ignore les fichiers temporaires ou cachés (ex: .tmp, .DS_Store)
            if event.src_path.split("/")[-1].startswith("."):
                return
                
            logger.info(f"Modification détectée : {event.src_path}")
            self.pending_event = True
            self.last_trigger_time = time.time()

    def on_created(self, event):
        self.process_event(event)

    def on_moved(self, event):
        # Important : capture aussi les fichiers déplacés dans le dossier
        self.process_event(event)

    def run_ingestion(self):
        """Relance l'ingestion via le script src/ingestion/main.py en mode Compléter."""
        logger.info("🚀 Déclenchement de l'ingestion incrémentale...")
        
        try:
            # Appel du script spécifique avec l'argument --mode c
            # On utilise sys.executable pour garantir l'utilisation du même venv
            subprocess.run(
                [sys.executable, str(self.ingestion_script), "--mode", "c"],
                check=True,
                cwd=str(config.BASE_DIR) # On définit le répertoire de travail à la racine
            )
            logger.info("✅ Pipeline d'ingestion terminé avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Échec de l'ingestion automatique (Code {e.returncode}).")
        except Exception as e:
            logger.error(f"❌ Erreur système lors du lancement : {e}")

def start_watching():
    """Point d'entrée principal du service de surveillance."""
    if not config.DATASET_DIR.exists():
        logger.error(f"Le dossier à surveiller n'existe pas : {config.DATASET_DIR}")
        return

    handler = DatasetHandler(debounce_seconds=7) # Augmenté légèrement pour les gros batchs
    observer = Observer()
    observer.schedule(handler, str(config.DATASET_DIR), recursive=True)
    observer.start()
    
    logger.info(f"👀 SmartSearch Watcher actif sur : {config.DATASET_DIR}")
    logger.info("En attente de nouveaux fichiers...")
    
    

    try:
        while True:
            # Mécanisme de Debouncing : on attend que le calme revienne
            if handler.pending_event:
                time_since_last = time.time() - handler.last_trigger_time
                if time_since_last > handler.debounce_seconds:
                    handler.run_ingestion()
                    handler.pending_event = False
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Arrêt du Watcher...")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    start_watching()