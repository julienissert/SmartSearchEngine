# src/search/main.py
from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="MAYbe Here - Search API",
    description="API de recherche multimodale Food & Medical",
    version="1.0.0"
)

app.include_router(router)