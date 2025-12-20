# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv  

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env") 

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL", "openai/clip-vit-base-patch32")

DATASET_DIR = BASE_DIR / "raw-datasets"
COMPUTED_DIR = BASE_DIR / "computed-data"
FAISS_INDEX_DIR = COMPUTED_DIR / "indexes"
METADATA_FILE = COMPUTED_DIR / "metadata_db.json"