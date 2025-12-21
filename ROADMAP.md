# Projet : SmartSearchEngine - ROADMAP.md

Ce document répertorie l'état d'avancement du moteur de recherche multimodal, les problématiques résolues et les pistes d'amélioration pour les prochaines itérations.

## État des Lieux (Acquis)


* **Complétion du Dataset (Mode incrémental)** : [OK] Le système propose désormais le mode `(R)éinitialiser` ou `(C)ompléter`. En mode complétion, le `IngestionService` identifie les fichiers déjà présents dans les métadonnées pour ne traiter que les nouveaux arrivants.
* **Recherche Hybride CLIP-OCR** : [OK] La recherche projette l'image et le texte OCR dans le même espace vectoriel pour une précision maximale. Pour gérer le volume de voisins, le `retriever` effectue une exploration large ($K=100$) avant d'appliquer un filtrage par consensus et une validation de label.
* **Segmentation par Domaine** : [OK] Les métadonnées et les index FAISS sont découpés physiquement par domaine (`food`, `medical`), garantissant une isolation des données et une performance accrue.

---

## Focus Technique : Optimisation de l'OCR

Le choix de l'OCR se décompose en deux niveaux décisionnels : le choix de la solution globale et le réglage fin du moteur Tesseract.

### Niveau 1 : Le choix de la Librairie (Quelle "marque" ?)
Nous comparons les trois solutions majeures du marché pour déterminer la brique principale du projet.

| Librairie | Avantages | Inconvénients |
| :--- | :--- | :--- |
| **Tesseract** | Très léger, pas de GPU requis, standard industriel. | Sensible à la qualité de l'image et aux rotations. |
| **EasyOCR** | Excellente gestion de l'écriture manuscrite. | Lent sur CPU, gourmand en RAM. |
| **PaddleOCR** | Précision SOTA (State of the Art), ultra-rapide. | Installation plus complexe (PaddlePaddle). |

### Niveau 2 : Configuration de Tesseract (Si choisi au Niveau 1)
Tesseract dispose de plusieurs modes de fonctionnement internes (OEM - OCR Engine Mode) :

1.  **Legacy (Mode 0)** : Reconnaissance de motifs à l'ancienne. Très rapide, mais obsolète pour les polices complexes.
2.  **LSTM Only (Mode 1)** : Utilise des réseaux de neurones. C'est le mode le plus précis pour le texte moderne et les documents denses.
3.  **Combined (Mode 2)** : Mélange les deux approches. C'est le mode le plus robuste mais le plus lent.

### Stratégie de Multi-traitement (Pipeline Pro)
Pour maximiser la fiabilité, nous envisageons une logique de "Cascade" :
* **Passage 1 (Rapide)** : Tesseract en mode LSTM.
* **Passage 2 (Expert)** : Si le score de confiance est bas (< 70%), déclenchement automatique de **PaddleOCR** pour une analyse profonde.
* **Impact** : Précision maximale pour le domaine médical au prix d'un temps d'ingestion accru.
---

## Évolutions Futures & Questions en Suspend

### 1. Couche RAG (Retrieval-Augmented Generation)
* **Objectif** : Transformer les résultats bruts en réponses naturelles.

### 2. Framework d'Évaluation (Benchmarking)
* **Question** : Comment mesurer mathématiquement la qualité des résultats ?
* **Action** : Mettre en place des métriques comme le **MAP (Mean Average Precision)** ou le **Hit Rate** sur un dataset de test.

### 3. Optimisation de l'Indexation Large Échelle
* **Problématique** : Actuellement, nous utilisons `IndexFlatL2` (recherche exhaustive).
* **Question** : À partir de quel volume de données devons-nous passer sur un index `IVFFlat` (indexation par grappes) pour rester sous les 100ms ?

### 4. Interface Utilisateur (UI)
* **Besoin** : Une interface (ex: Streamlit) pour uploader une photo et visualiser les images de `visual_confirmation` côte à côte avec les fiches `enriched_info`.
