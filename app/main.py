# app/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from .analyzer import analyze_image
from .retriever import retriever
from .composer import detect_domain_from_matches, compose_result
from .embed import embed_text
from .storage import save_image_bytes
import io
from PIL import Image
import numpy as np
import json

app = FastAPI(title="Simple Image -> Dataset Search")

@app.post("/ingest")
async def ingest_csv(file: UploadFile = File(...), domain: str = Form(...), text_field: str = Form(...), name_field: str = Form(None), calories_field: str = Form(None)):
    """
    Ingest a CSV file (CSV must have columns matching text_field and optional name_field/calories_field).
    Example: text_field='description', name_field='name', calories_field='calories_per_100g'
    """
    import pandas as pd
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    items = []
    for _, row in df.iterrows():
        text = str(row[text_field])
        name = str(row[name_field]) if name_field and name_field in df.columns else None
        calories = None
        if calories_field and calories_field in df.columns:
            try:
                calories = float(row[calories_field])
            except Exception:
                calories = None
        emb = embed_text(text)
        items.append((emb, {"id": f"{domain}/{len(retriever.metadatas)+len(items)}", "domain": domain, "name": name, "calories_per_100g": calories, "text": text}))
    if items:
        embs = np.vstack([e for e, m in items]).astype("float32")
        metas = [m for e, m in items]
        retriever.add(embs, metas)
    return {"ingested": len(items)}

@app.post("/search")
async def search(image: UploadFile = File(...)):
    image_bytes = await image.read()
    save_path = save_image_bytes(image_bytes, image.filename)
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    analysis = analyze_image(pil)
    img_emb = analysis["image_embedding"]
    labels = analysis["labels"]
    ocr_text = analysis["ocr_text"]

    # First: search image embedding against index (we used textual embeddings in same CLIP space)
    matches = retriever.search(img_emb, k=10)

    # Determine domain majority (e.g., "food")
    domain = detect_domain_from_matches(matches)

    # For domain 'food', we query again but can refine:
    # build a combined query: prefer OCR text if present
    query_emb = img_emb
    if ocr_text:
        # combine image + ocr text embeddings (average)
        txt_emb = embed_text(ocr_text)
        query_emb = (img_emb + txt_emb) / 2.0
        norm = np.linalg.norm(query_emb)
        if norm>0:
            query_emb = query_emb / norm

    domain_filtered = [m for m in retriever.search(query_emb, k=20) if m["meta"].get("domain")==domain]

    # Compose final result
    result = compose_result(labels, domain_filtered[:5])

    return JSONResponse({
        "saved_image": save_path,
        "analysis": {
            "labels": labels,
            "ocr_text": ocr_text
        },
        "domain": domain,
        "matches": domain_filtered[:5],
        "result": result
    })

# quick endpoint to check index size
@app.get("/status")
def status():
    return {"indexed_items": len(retriever.metadatas)}