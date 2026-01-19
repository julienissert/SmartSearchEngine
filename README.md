# SmartSearchEngine - Moteur Multimodal Unifié

Moteur de recherche intelligent capable de classifier et de retrouver des informations à partir de requêtes par image ou texte.

## Architecture du Projet

Le système sépare strictement les données brutes, les artefacts calculés et la logique métier :

* **`raw-datasets/`** : Données sources (CSV, PDF, Images, TXT, H5).
* **`computed-data/`** : Fichiers générés (Index FAISS et base JSON).
    * **`indexes/`** : Contient les fichiers `.index` par domaine.
    * **`metadata_db.json`** : Base de données textuelle complète synchronisée.
* **`src/`** : Dossier racine du code source.

## Installation des dépendances :
```bash
pip install -r requirements.txt
```

## Guide d'Utilisation
L'ensemble du projet se pilote via l'orchestrateur unique à la racine du dossier src/.

1. Ingestion des données (Construction)
```bash
python src/main.py ingest
```
2. Lancement du Serveur (Service API)
```bash
python src/main.py serve
```
Le serveur sera disponible sur http://localhost:8000/docs. 