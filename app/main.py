"""
Main FastAPI Application Entry Point.
Initializes lifespan, mounts REST endpoints, and serves the enterprise web client.
"""
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router
from app.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION
from app.services.search_service import SearchEngine

logger = logging.getLogger("nexora.main")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan:
    Preloads Gensim models, TF-IDF matrices, and 50,000 documents into memory
    before accepting search requests to ensure sub-10ms response latencies.
    """
    logger.info("Initializing search engine assets on startup...")
    engine = SearchEngine.get_instance()
    engine.initialize()
    logger.info("Semantic Search Service is warmed up and ready to serve traffic.")
    yield
    logger.info("Shutting down Semantic Search Service.")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for cross-origin or local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    """Serves the primary web client application."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(str(index_file))
