
# src/config.py
import os

EMBEDDING_DIM = 512
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "raw-datasets")
COMPUTED_DIR = os.path.join(BASE_DIR, "computed-data")

FAISS_INDEX_DIR = os.path.join(COMPUTED_DIR, "indexes")
METADATA_FILE = os.path.join(COMPUTED_DIR, "metadata_db.json")

TARGET_DOMAINS = ["food", "medical"]

os.makedirs(FAISS_INDEX_DIR, exist_ok=True)