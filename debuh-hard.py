# debug_hard.py
import sys
import os
import logging

# On configure les logs pour tout voir
logging.basicConfig(level=logging.DEBUG)

print("\n--- 1. TEST DES IMPORTS ---")
try:
    import numpy as np
    print("✅ Numpy importé")
    from PIL import Image
    print("✅ PIL (Pillow) importé")
    from paddleocr import PaddleOCR
    print("✅ PaddleOCR importé")
except ImportError as e:
    print(f"❌ ERREUR CRITIQUE D'IMPORT : {e}")
    sys.exit(1)

# Chemin exact de ton image qui a échoué
IMG_PATH = "/Users/julienissert/Documents/Dev/ENSIM/SmartSearchEngine/raw-datasets/Food Classification dataset/apple_pie/134.jpg"

print(f"\n--- 2. CHARGEMENT DE L'IMAGE : {os.path.basename(IMG_PATH)} ---")
try:
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Le fichier n'existe pas : {IMG_PATH}")
        
    img = Image.open(IMG_PATH).convert("RGB")
    img_array = np.array(img)
    print(f"✅ Image chargée. Dimensions : {img.size}")
except Exception as e:
    print(f"❌ ERREUR LECTURE IMAGE : {e}")
    sys.exit(1)

print("\n--- 3. INITIALISATION DU MOTEUR (C'est souvent ici que ça casse sur Mac) ---")
try:
    # Paramètres identiques à src/ingestion/loaders/image_loader.py [cite: 66]
    # Mais avec show_log=True pour voir ce qui se passe
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='fr')
    print("✅ Moteur initialisé")
except Exception as e:
    print("❌ ERREUR INITIALISATION PADDLE :")
    print(e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- 4. TENTATIVE D'EXTRACTION ---")
try:
    result = ocr_engine.ocr(img_array)
    print("✅ RÉSULTAT OBTENU :")
    print(result)
except Exception as e:
    print("❌ ERREUR LORS DE L'OCR :")
    print(e)
    import traceback
    traceback.print_exc()