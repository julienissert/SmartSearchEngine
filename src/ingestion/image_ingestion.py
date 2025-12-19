#ingestion/image_ingestion.py
from PIL import Image
import pytesseract

def load_image(path):
    img = Image.open(path)
    ocr_text = pytesseract.image_to_string(img)

    return [{
        "source": path,
        "type": "image",
        "image": img,
        "content": ocr_text
    }]
