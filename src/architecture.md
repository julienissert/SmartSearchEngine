datasets/
├── drugs-list/
├── fast-food-nutrition/
├── food-101/
├── food-nutrition/
scripts/
│
├── ingestion/
│   ├── csv_ingestion.py
│   ├── pdf_ingestion.py
│   ├── txt_ingestion.py
│   ├── image_ingestion.py
│   ├── h5_ingestion.py
│   ├── folder_scanner.py      # parcours dossier
│   └── dispatcher.py          # sélection du loader selon le fichier
│
├── embeddings/
│   ├── text_embeddings.py     # embed_text()
│   └── image_embeddings.py    # embed_image()
│
├── indexing/
│   ├── faiss_index.py         # index vectoriel
│   └── metadata_index.py      # heuristiques de domaine / nettoyage
│
├── utils/
│   ├── preprocessing.py       # clean texte
│   ├── label_detector.py       # clean texte
│   └── domain_detector.py     # détecte domaine food/medical
│
├── config.py
└── main.py                    # pipeline d’ingestion complet
