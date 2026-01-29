# src/utils/spinner.py
import threading
import time
import itertools

class TqdmHeartbeat:
    """Anime la description d'une barre tqdm pour prouver que le programme vit."""
    def __init__(self, pbar, base_message="Ingestion"):
        self.pbar = pbar
        self.base_message = base_message
        self.symbols = itertools.cycle(['|', '/', '-', '\\'])
        self.running = False
        self.thread = None

    def _animate(self):
        while self.running:
            sym = next(self.symbols)
            self.pbar.set_description(f"{sym} {self.base_message}")
            time.sleep(0.5)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.pbar.set_description(self.base_message)