# SmartSearch - Système RAG Multimodal

**SmartSearch** est un moteur d'ingestion et de recherche documentaire intelligent. Il permet d'indexer des datasets hétérogènes (PDF, CSV, JSON, Images, H5) pour alimenter un système **RAG (Retrieval-Augmented Generation)**.

La fonctionnalité clé du système est sa capacité **multimodale** : il permet, à partir d'une **photo en entrée**, de retrouver l'élément correspondant et toutes les informations textuelles ou techniques liées existant dans la base de connaissances.

---

## Structure du Code

L'architecture est modulaire, séparant l'ingestion, la vectorisation (embeddings) et la recherche.

```bash
.
├── raw-datasets/           # Dossier surveillé (Point d'entrée des fichiers)
├── src/
│   ├── config.py           # Configuration (Variables d'env, chemins)
│   ├── main.py             # Point d'entrée CLI
│   ├── ingestion/          # Pipeline de chargement
│   │   ├── dispatcher.py   # Routeur vers le bon loader
│   │   ├── folder_scanner.py
│   │   └── loaders/        # Parsers (pdf, image, csv, json, h5, txt...)
│   ├── embeddings/         # Moteurs de vectorisation
│   │   ├── image_embeddings.py # Modèle Vision (CLIP)
│   │   └── text_embeddings.py  # Modèle NLP
│   ├── indexing/           # Gestion de l'index vectoriel (FAISS)
│   ├── search/             # Logique de recherche (Retriever)
│   ├── interface/          # Dashboard de visualisation
│   └── utils/              # Utilitaires (Watcher, Logger)
├── dockerfile              # Image compatible CPU/CUDA
├── docker-compose.yml      # Orchestration
└── .env                    # Variables d'environnement
```

---

## Installation & Exécution (Local)

1. Prérequis
- Python 3.10+
- (Optionnel) CUDA Toolkit (si GPU NVIDIA disponible)

2. Configuration
Créez un fichier .env à la racine :

```bash
TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2
IMAGE_MODEL=openai/clip-vit-base-patch32
```

3. Installation
```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
```

```bash
# Installer les dépendances
pip install -r requirements.txt
```

4. Lancer l'Ingestion
Vous avez deux modes pour ingérer des données :

- Option A : Ingestion Manuelle (One-shot) Pour scanner et indexer tout le dossier raw-datasets immédiatement en réinitalisant toutes les bases de données existantes :
```bash
python -m src.main ingest -m c
```

- Option B : Ingestion Manuelle (One-shot) Pour scanner et indexer tout le dossier raw-datasets immédiatement en complétant les bases de données existantes avec le contenu des nouveaux datasets :
```bash
python -m src.main ingest -m c
```

- Option C : Mode Surveillance (Watcher) Le système surveille le dossier en temps réel. Déposez un fichier, il est ingéré automatiquement.
```bash
python -m src.utils.watcher
```

5. Lancer la Recherche / Interface
Pour démarrer le dashboard :

```bash
python -m src.interface.dashboard
```

---

## Installation & Exécution (via Docker)

## 1. Préparation des Données

Avant de lancer les conteneurs, vous devez alimenter le système avec vos fichiers.

1. Assurez-vous que le dossier `raw-datasets` existe à la racine du projet (là où se trouve le fichier `docker-compose.yml`).
2. **Déposez vos fichiers** (Images, PDF, CSV, JSON) dans ce dossier.

> **Note :** Ce dossier est monté en tant que volume dans les conteneurs (`/app/raw-datasets`). Toute modification locale est immédiatement visible par le système Docker.

## 2. Configuration Matérielle (GPU vs CPU)

Le fichier `docker-compose.yml` est configuré par défaut pour utiliser l'accélération **NVIDIA GPU** (CUDA).

### Option A : Avec GPU NVIDIA (Recommandé)

Aucune modification n'est nécessaire. Le système utilisera le GPU pour :

* Les embeddings d'images (CLIP).
* L'OCR.
* Le LLM (Ollama).

### Option B : Sans GPU (CPU Only)

Si vous n'avez pas de carte graphique dédiée, vous devez **modifier le fichier `docker-compose.yml**` pour désactiver les réservations de ressources GPU.

**Modifications à effectuer :**
Commentez (ajoutez `#` au début de la ligne) les blocs `deploy` dans :

*Exemple de section commentée :*

```yaml
  # deploy:
  #   resources:
  #     reservations:
  #       devices:
  #         - driver: nvidia
  #           count: 1
  #           capabilities: [gpu]
```


- Option A : Ingestion Manuelle (One-shot) Pour scanner et indexer tout le dossier raw-datasets immédiatement en réinitalisant toutes les bases de données existantes:
```bash
docker-compose run --rm ingest ingest -m r
```

- Option B : Ingestion Manuelle (One-shot) Pour scanner et indexer tout le dossier raw-datasets immédiatement en complétant les bases de données existantes avec le contenu des nouveaux datasets.
```bash
docker-compose run --rm ingest ingest -m c
```


**2. Ingestion Automatique (Watcher)**
Pour lancer le service qui surveille le dossier `raw-datasets` en continu et ingère les nouveaux fichiers automatiquement :

```bash
docker compose up -d watcher

```

### C. Lancer l'API de Recherche

Pour démarrer le serveur API (disponible sur `http://localhost:8000`) :

```bash
docker compose up -d search

```

## 4. Vérification et Logs

Pour vérifier que tout fonctionne correctement ou diagnostiquer une erreur :

```bash
# Voir les statuts des conteneurs
docker compose ps

# Voir les logs d'un service spécifique (ex: watcher)
docker compose logs -f watcher

```

## 5. Maintenance

* **Arrêter les services :** `docker compose down`
* **Mettre à jour l'image :** `docker compose pull`.