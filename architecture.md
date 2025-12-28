# Architecture du Projet - SmartSearchEngine

L'ensemble du système est conçu selon le principe de **Séparation des Préoccupations (SoC)** pour garantir la scalabilité et la performance industrielle lors du traitement de jeux de données volumineux.

## Arborescence du Projet

```text
SmartSearchEngine/
├── raw-datasets/               # Données sources (Fichiers bruts à ingérer : CSV, PDF, Images, H5)
├── computed-data/              # Persistance des données générées par le pipeline
│   ├── indexes/                # Index vectoriels FAISS (.index) segmentés par domaine
│   ├── metadata/               # Bases de métadonnées JSON (.json) par domaine
│   └── visualizations/         # Explorateurs 3D interactifs générés (ex: space_explorer.html)
│
└── src/                        # Racine du code source
    ├── main.py                 # ORCHESTRATEUR GÉNÉRAL : CLI pour lancer 'ingest' ou 'serve'.
    ├── config.py               # CONFIGURATION CENTRALE : Matériel (CPU/CUDA/MPS), modèles et seuils.
    │
    ├── ingestion/              # --- MODULE INGESTION (CONSTRUCTION) ---
    │   ├── main.py             # Point d'entrée CLI local pour l'ingestion de données.
    │   ├── service.py          # Gestionnaire de workflow : Gère l'orchestration et le Batching.
    │   ├── core.py             # Logique métier : Vectorisation groupée et classification finale.
    │   ├── dispatcher.py       # Routeur de fichiers : Associe chaque extension au loader adéquat.
    │   ├── folder_scanner.py   # Scan récursif intelligent du dossier source.
    │   └── loaders/            # LOGIQUE D'EXTRACTION PAR FORMAT
    │       ├── base_loader.py  # Interface abstraite définissant le contrat des loaders.
    │       ├── csv_loader.py   # Chargeur haute performance utilisant Pandas.
    │       ├── image_loader.py # Chargeur d'images avec extraction de texte par OCR.
    │       ├── pdf_loader.py   # Extracteur de texte pour documents PDF.
    │       ├── txt_loader.py   # Extracteur de textes bruts et détection de listes.
    │       └── h5_loader.py    # Chargeur optimisé pour fichiers de données structurées H5.
    │
    ├── search/                 # --- MODULE SEARCH (SERVICE API) ---
    │   ├── main.py             # Lancement du serveur FastAPI avec gestion du cycle de vie.
    │   ├── routes.py           # Définition des endpoints API (ex: /search).
    │   ├── processor.py        # Analyseur de requête : Transforme l'image reçue en vecteur + OCR.
    │   ├── retriever.py        # Moteur de recherche hybride multi-domaine (FAISS).
    │   └── composer.py         # Formateur de réponse : Synthèse des métadonnées et scores.
    │
    ├── embeddings/             # --- INTELLIGENCE & VECTORISATION ---
    │   ├── image_embeddings.py # Modèle CLIP Vision : Supporte l'inférence par lots (Batch).
    │   └── text_embeddings.py  # Modèle CLIP Text : Supporte l'inférence par lots (Batch).
    │
    ├── indexing/               # --- COUCHE DE PERSISTANCE (RAM-FIRST) ---
    │   ├── faiss_index.py      # Gestion des index FAISS en RAM avec sauvegarde différée.
    │   └── metadata_index.py   # Gestion de la base JSON segmentée pour des accès rapides.
    │
    └── utils/                  # --- OUTILS, HEURISTIQUES & ANALYSE ---
        ├── domain_detector.py  # Classification Zero-Shot Food/Medical optimisée (One-Shot Vector).
        ├── label_detector.py   # Analyseur de structure pour la découverte automatique des labels.
        ├── handlers/           # LOGIQUE DE RÉSOLUTION DES ÉTIQUETTES
        │   ├── raw_handler.py  # Détecteur de labels pour fichiers non structurés (Image/PDF).
        │   └── structured_handler.py # Stratégie hybride pour données tabulaires (CSV/H5).
        ├── environment.py      # Vérificateur système : GPU, Tesseract et arborescence.
        ├── preprocessing.py    # Nettoyage sémantique et normalisation du texte.
        ├── visualizer.py       # Générateur de nuages de points 3D (Plotly + PCA).
        └── logger.py           # Système de logs industriel avec rotation et compression Gzip.

```