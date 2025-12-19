# src/search/routes.py
from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
from .processor import analyze_query
from .retriever import retriever
from .composer import composer

router = APIRouter()

@router.post("/search")
async def search_endpoint(image: UploadFile = File(...)):
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 1. Analyse
    vector, ocr_text = analyze_query(pil_img)
    
    # 2. Recherche vectorielle
    matches = retriever.search(vector, k=5)
    
    # 3. Composition (Plus besoin de refresh() ici, gain de performance !)
    response = composer.build_response(matches, ocr_text)
    
    return response