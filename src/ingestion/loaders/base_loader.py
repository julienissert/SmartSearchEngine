# src/ingestion/loaders/base_loader.py
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    @abstractmethod
    def get_supported_extensions(self) -> list:
        """Renvoie la liste des extensions gérées par ce loader."""
        pass

    @abstractmethod
    def can_handle(self, extension: str) -> bool:
        """Vérifie si ce loader peut traiter l'extension donnée."""
        pass

    @abstractmethod
    def load(self, path: str, valid_labels=None) -> list:
        """Logique d'extraction des données."""
        pass