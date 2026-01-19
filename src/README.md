# Documentation Technique - Dossier Source (`src`)

Ce dossier contient l'ensemble du code source du moteur d'ingestion et de recherche multimodal. Cette documentation détaille l'organisation des modules et le fonctionnement interne du pipeline.

## Organisation des Modules

Le code est architecturé selon le principe de **Séparation des Préoccupations (SoC)**. Chaque sous-dossier a une responsabilité unique.

### 1. `ingestion/` (La couche d'entrée)
Responsable de la lecture brute des fichiers.
* **`dispatcher.py`** : Le "routeur". Il analyse l'extension du fichier et appelle le loader approprié.
* **`folder_scanner.py`** : Parcourt récursivement le dossier datasets.
* **Loaders Spécialisés** :
    * `csv_ingestion.py` : Découpe chaque ligne en un document distinct. Tente de détecter une colonne "Item".
    * `h5_ingestion.py` : Utilise un *Lazy Loading* pour ne charger que la structure des fichiers H5 volumineux (>10 Mo).
    * `image_ingestion.py` : Utilise Tesseract pour l'OCR.
    * `pdf_ingestion.py` & `txt_ingestion.py` : Extraction de texte brut.

### 2. `utils/` (L'Intelligence)
Contient les heuristiques et la logique métier.
* **`label_detector.py`** : Cerveau de la classification.
    * Analyse statistique des dossiers pour ignorer le bruit
    * Détecte les listes de référence dans les fichiers `.txt`.
    * Combine Dossier + Texte + Vision pour déduire un label.
* **`domain_detector.py`** : Classifie chaque document en `food` ou `medical` via comparaison vectorielle (Zero-Shot Classification).
* **`preprocessing.py`** : Nettoyage basique des chaînes de caractères (minuscules, suppression caractères spéciaux).

### 3. `embeddings/` (La Vectorisation)
Transforme les données en vecteurs mathématiques 
* **Modèle** : CLIP (`openai/clip-vit-base-patch32`) est utilisé pour **le texte ET l'image** afin de garantir un espace vectoriel commun.
* **`text_embeddings.py`** : Tronque le texte à 77 tokens (limite hardware de CLIP).
* **`image_embeddings.py`** : Redimensionne et normalise les images pour le modèle Vision Transformer.

### 4. `indexing/` (Le Stockage)
Gère la persistance des données.
* **`faiss_index.py`** : Gère les index vectoriels `.index` (recherche de similarité rapide). Crée un index séparé par domaine (Food/Medical).
* **`metadata_index.py`** : Gère la base de données JSON (`metadata_db.json`). Assigne un ID unique séquentiel à chaque document.

---

## Fichiers Principaux

### `main.py` (L'Orchestrateur)
C'est le point d'entrée unique. Il exécute le pipeline séquentiel :
1.  **Initialisation** : Vérification des chemins.
2.  **Apprentissage (Warm-up)** : Appelle `analyze_dataset_structure` pour découvrir les labels valides avant de commencer.
3.  **Boucle de Traitement** :
    * Scan des fichiers.
    * Dispatch vers le bon loader.
    * Détection Domaine + Label.
    * Calcul Embedding.
    * Stockage.
4.  **Finalisation** : Sauvegarde du JSON sur le disque.

### `config.py` (Configuration)
Centralise les constantes du projet :
* `EMBEDDING_DIM` : 512 (Fixé par l'architecture CLIP).
* Chemins des dossiers (`../datasets`, `./vector_indexes`).
* Noms des modèles Hugging Face.

---

### Exécution
Le script doit toujours être lancé depuis la **racine du projet** pour que les imports Python fonctionnent :
```bash

python -m src.main [COMMAND]

```

| Action            | Commande                          | Description                                                                 |
|-------------------|-----------------------------------|-----------------------------------------------------------------------------|
| Réinitialisation  | `python -m src.main ingest -m r`   | Efface les index existants et reconstruit tout à partir de zéro.            |
| Complétion        | `python -m src.main ingest -m c`   | Ajoute uniquement les nouveaux fichiers détectés dans le dossier dataset sans effacer l'existant. |
| Surveillance      | `python -m src.main watch`         | Lance le service Watcher qui automatise l'ingestion dès qu'un fichier est ajouté ou déplacé. |
| Serveur API       | `python -m src.main serve`         | Démarre l'API FastAPI pour effectuer des recherches (disponible sur le port 8000). |
