# SmartSearchEngine - Moteur Multimodal Unifié

Moteur de recherche intelligent capable de classifier et de retrouver des informations à partir de requêtes par image ou texte.

## Architecture du Projet

Le système sépare strictement les données brutes, les artefacts calculés et la logique métier :

* **`raw-datasets-01/`** : Données sources (CSV, PDF, Images, TXT, H5).
* **`computed-data-03/`** : Fichiers générés (Index FAISS et base JSON).
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
python src/ingestion/main.py --mode c (Pour complétion de données précédement compilées.)
python src/ingestion/main.py --mode r (Pour effacement et reconstruction des données compilées.)
```
2. Lancement du Serveur (Service API)
```bash
python src/main.py serve
```
Le serveur sera disponible sur http://localhost:8000. L'endpoint principal est /search

3. Lancement du serveur d'ingestion (Watchdog)
```bash
python -m src.utils.watcher
````
Celui sert à lancer l'ingestion automatiquement dès l'ajout de nouveaux datasets dans le répertoire dédié.

