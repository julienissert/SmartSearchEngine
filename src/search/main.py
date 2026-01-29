from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.search.routes import router
from src.utils.logger import setup_logger

logger = setup_logger("SearchAPI")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(" L'API de recherche est prête et opérationnelle.")
    logger.info(" Accédez à l'interface de test : http://localhost:8000/docs")
    yield
    logger.info(" Arrêt de l'API de recherche.")

app = FastAPI(
    title="MAYbe Here - Search API",
    description="API de recherche multimodale Food & Medical",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)