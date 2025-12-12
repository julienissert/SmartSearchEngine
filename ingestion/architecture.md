datasets/
├── drugs-list/
├── fast-food-nutrition/
├── food-101/
src/
│
├── ingestion/
│   ├── csv_ingestion.py
│   ├── pdf_ingestion.py
│   ├── txt_ingestion.py
│   ├── image_ingestion.py
│   ├── h5_ingestion.py
│   ├── folder_scanner.py      
│   └── dispatcher.py          
│
├── embeddings/
│   ├── text_embeddings.py     
│   └── image_embeddings.py    
│
├── indexing/
│   ├── faiss_index.py         
│   └── metadata_index.py     
│
├── utils/
│   ├── preprocessing.py       
│   ├── label_detector.py     
│   └── domain_detector.py    
│
├── config.py
└── main.py                   
