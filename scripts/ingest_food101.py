# scripts/ingest_food101.py

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.append(ROOT)
import json
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from app.retriever import retriever
import clip
model, preprocess = clip.load("ViT-B/32")

device = "cuda" if torch.cuda.is_available() else "cpu"

class Food101Ingestor:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir

        # Load CLIP model
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def embed_image(self, img_path: str):
        """Returns a CLIP embedding for one image."""
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return None

        image_tensor = self.preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding.cpu().numpy()[0]

        return embedding

    def ingest(self):
        images_root = os.path.join(self.dataset_dir, "images")
        metadata = []

        print("📥 Starting FOOD-101 ingestion...")

        for class_name in sorted(os.listdir(images_root)):
            class_dir = os.path.join(images_root, class_name)

            if not os.path.isdir(class_dir):
                continue

            print(f"➡️ Processing class: {class_name}")

            for img_file in tqdm(os.listdir(class_dir)):
                img_path = os.path.join(class_dir, img_file)

                # Compute embedding
                embedding = self.embed_image(img_path)
                if embedding is None:
                    continue

                # Add to FAISS
                retriever.add_vector(
                    vector=embedding,
                    metadata={
                        "label": class_name.replace("_", " "),
                        "image_path": img_path,
                        "domain": "food.images",
                        "type": "food_item"
                    }
                )

                # Also save metadata for traceability
                metadata.append({
                    "label": class_name.replace("_", " "),
                    "image_path": img_path,
                    "domain": "food.images"
                })

        # Save metadata file
        with open("stores/food101_metadata.jsonl", "w") as f:
            for m in metadata:
                f.write(json.dumps(m) + "\n")

        print("✅ FOOD-101 ingestion completed!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_food101.py path/to/food-101")
        exit()

    dataset_path = sys.argv[1]
    ingestor = Food101Ingestor(dataset_path)
    ingestor.ingest()