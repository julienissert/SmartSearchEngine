# app/storage.py
from pathlib import Path
import uuid
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def save_image_bytes(b: bytes, filename: str = None) -> str:
    if filename is None:
        filename = f"{uuid.uuid4().hex}.jpg"
    path = IMAGES_DIR / filename
    with open(path, "wb") as f:
        f.write(b)
    return str(path)