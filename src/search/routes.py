# src/search/routes.py
from fastapi import APIRouter, File, UploadFile
from PIL import Image
import io
from src.search.processor import analyze_query
from src.search.retriever import retriever
from src.search.composer import composer

router = APIRouter()

@router.post("/search")
async def search_endpoint(image: UploadFile = File(...)):
    # 1. Lecture de l'image
    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # 2. ANALYSE 
    processed_query = analyze_query(pil_img)
    
    # 3. RECHERCHE 
    matches = retriever.search(processed_query, k=5)
    
    # 4. COMPOSITION DE LA RÉPONSE FACTUELLE
    response = composer.build_response(matches, processed_query["ocr_text"])
    
    return response