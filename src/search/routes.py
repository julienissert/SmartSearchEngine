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
    
    image_vector, ocr_text = analyze_query(pil_img)
    matches = retriever.search(image_vector, query_ocr=ocr_text, k=5)
    response = composer.build_response(matches, ocr_text)
    
    return response