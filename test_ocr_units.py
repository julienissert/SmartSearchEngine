# test_ocr_unit.py
import sys
from pathlib import Path
import logging

# Configuration du chemin pour simuler l'exécution depuis la racine
SRC_DIR = Path("src").resolve()
sys.path.append(str(SRC_DIR))

# On configure le logger pour voir les sorties de Paddle
from src.utils.logger import setup_logger
logger = setup_logger("TestOCR")

def test_single_image(image_path):
    print(f"--- Test sur : {image_path} ---")
    
    # Importation à l'intérieur pour vérifier les dépendances
    try:
        from src.ingestion.loaders.image_loader import ImageLoader
        loader = ImageLoader()
        
        # Vérification de l'extension
        if not loader.can_handle(Path(image_path).suffix):
            print("❌ Extension non supportée par ImageLoader")
            return

        # Simulation du chargement (comme le fait le worker)
        # [cite: 67] Charge l'image et [cite: 68] lance l'OCR
        results = loader.load(image_path)
        
        if not results:
            print("❌ Aucun résultat retourné (Erreur interne ?)")
        else:
            doc = results[0]
            ocr_content = doc.get("content", "")
            print("✅ OCR Succès !")
            print(f"📄 Texte extrait ({len(ocr_content)} chars) :")
            print(f"   '{ocr_content[:100]}...'") # Affiche les 100 premiers chars
            
    except Exception as e:
        print(f"❌ Crash du test : {e}")

if __name__ == "__main__":
    # Remplace par un chemin réel vers une image de ton dataset
    test_single_image("/Users/julienissert/Documents/Dev/ENSIM/SmartSearchEngine/raw-datasets/Food Classification dataset/apple_pie/134.jpg")