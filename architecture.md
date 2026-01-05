# Architecture du Projet - SmartSearchEngine

L'ensemble du système est conçu selon le principe de **Séparation des Préoccupations (SoC)** pour garantir la scalabilité et la performance industrielle lors du traitement de jeux de données volumineux.

## Arborescence du Projet

```text
SmartSearchEngine/
├── raw-datasets/               # Données sources : fichiers bruts à ingérer (CSV, PDF, Images, H5)
├── computed-data/              # Persistance : stockage des données générées par le pipeline
│   ├── indexes/                # Index vectoriels FAISS (.index) segmentés par domaine (Food/Medical)
│   ├── metadata/               # Bases de métadonnées JSON (.json) associées aux IDs des index
│   └── visualizations/         # Explorateurs 3D interactifs (HTML) de l'espace vectoriel
│
└── src/                        # Racine du code source
    ├── main.py                 # ORCHESTRATEUR GÉNÉRAL : CLI pour piloter l'ingestion ou le serveur API
    ├── config.py               # CONFIGURATION CENTRALE : Gestion du matériel (GPU/MPS/CPU) et des seuils IA
    │
    ├── ingestion/              # --- MODULE INGESTION (CONSTRUCTION) ---
    │   ├── main.py             # Point d'entrée CLI pour lancer ou compléter l'indexation
    │   ├── service.py          # Gestionnaire de workflow : orchestration du traitement par lots (Batching)
    │   ├── core.py             # Logique métier : coordination de la vectorisation massive et du stockage
    │   ├── dispatcher.py       # Routeur de fichiers : associe chaque extension au loader spécialisé adéquat
    │   ├── folder_scanner.py   # Scan récursif : identifie tous les fichiers valides dans les jeux de données
    │   └── loaders/            # LOGIQUE D'EXTRACTION PAR FORMAT
    │       ├── base_loader.py  # Interface abstraite définissant le contrat technique des loaders
    │       ├── csv_loader.py   # Chargeur CSV : transforme les lignes de données tabulaires en documents
    │       ├── image_loader.py # Chargeur d'images : extrait le texte via OCR (Tesseract)
    │       ├── pdf_loader.py   # Extracteur PDF : récupère le contenu textuel complet des documents
    │       ├── txt_loader.py   # Extracteur TXT : traite les textes bruts et détecte les listes de référence
    │       └── h5_loader.py    # Chargeur H5 : gère les données structurées volumineuses en mode "Lazy Loading"
    │
    ├── search/                 # --- MODULE SEARCH (SERVICE API) ---
    │   ├── main.py             # Serveur FastAPI : point d'entrée de l'application de recherche
    │   ├── routes.py           # Endpoints API : définition des routes (ex: /search) et réception des fichiers
    │   ├── processor.py        # Analyseur de requête : convertit l'image reçue en vecteur et extrait le texte OCR
    │   ├── retriever.py        # Moteur hybride : recherche par similarité vectorielle et mots-clés OCR
    │   └── composer.py         # Formateur de réponse : fusionne les matches vectoriels avec les métadonnées
    │
    ├── embeddings/             # --- INTELLIGENCE & VECTORISATION ---
    │   ├── image_embeddings.py # Modèle CLIP Vision : transforme les images en vecteurs mathématiques (512d)
    │   └── text_embeddings.py  # Modèle CLIP Text : transforme le texte en vecteurs dans le même espace que l'image
    │
    ├── indexing/               # --- COUCHE DE PERSISTANCE (RAM-FIRST) ---
    │   ├── faiss_index.py      # Gestionnaire FAISS : recherche de similarité rapide en RAM et sauvegarde disque
    │   └── metadata_index.py   # Index de métadonnées : gestionnaire CRUD pour les fichiers JSON de stockage
    │
    └── utils/                  # --- OUTILS, HEURISTIQUES & ANALYSE ---
        ├── domain_detector.py  # Classificateur Zero-Shot : oriente les documents vers le domaine Food ou Medical
        ├── label_detector.py   # Apprentissage de structure : découvre les labels valides (dossiers, fichiers TXT)
        ├── handlers/           # LOGIQUE DE RÉSOLUTION DES ÉTIQUETTES
        │   ├── raw_handler.py  # Détecteur pour documents bruts (Images/PDF) basé sur le contexte et CLIP
        │   └── structured_handler.py # Stratégie hybride pour données tabulaires (en-têtes, profilage statique)
        ├── environment.py      # Validateur système : vérifie le matériel IA et l'installation de Tesseract
        ├── preprocessing.py    # Nettoyage textuel : normalisation (minuscules, retrait caractères spéciaux)
        ├── visualizer.py       # Visualisation 3D : réduit les dimensions via PCA pour l'exploration interactive
        └── logger.py           # Système de logs : journalisation industrielle avec rotation et compression Gzip
```