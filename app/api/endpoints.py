"""
FastAPI REST API Endpoints for Semantic Similarity Search.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.models.schemas import (
    SearchRequest,
    SearchResponse,
    CompareRequest,
    CompareResponse,
    DocumentDetail,
    SystemStats,
)
from app.services.search_service import SearchEngine, get_search_engine

router = APIRouter(prefix="/api", tags=["Semantic Search"])


@router.get("/health", summary="Service Health Check")
def health_check(engine: SearchEngine = Depends(get_search_engine)):
    """Checks service availability and model initialization status."""
    return {
        "status": "healthy" if engine.is_loaded else "initializing",
        "models_ready": engine.is_loaded,
        "total_documents": engine.total_docs,
    }


@router.get("/stats", response_model=SystemStats, summary="System & Model Statistics")
def get_stats(engine: SearchEngine = Depends(get_search_engine)):
    """Returns vocabulary sizes, model dimensions, and indexed document counts."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search Engine models are still loading into memory.",
        )
    return engine.get_system_stats()


@router.get("/categories", response_model=List[str], summary="List Document Categories")
def get_categories(engine: SearchEngine = Depends(get_search_engine)):
    """Returns unique category tags for faceted search filtering."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search Engine models are still loading.",
        )
    return engine.categories


@router.post("/search", response_model=SearchResponse, summary="Execute Semantic Similarity Search")
def execute_search(
    request: SearchRequest,
    engine: SearchEngine = Depends(get_search_engine),
):
    """
    Primary Search Endpoint:
    - Preprocesses query text
    - Identifies OOV words requiring FastText character n-gram fallback
    - Calculates dense semantic embeddings
    - Computes cosine similarity across 50,000 articles
    - Returns ranked Top-K documents with similarity scores and attribution metadata
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search Engine is warming up. Please retry in a few seconds.",
        )
    try:
        return engine.search(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k,
            category=request.category,
            min_score=request.min_score,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing semantic search: {str(e)}",
        )


@router.post("/compare", response_model=CompareResponse, summary="Compare All Retrieval Models")
def compare_models(
    request: CompareRequest,
    engine: SearchEngine = Depends(get_search_engine),
):
    """
    Model Comparison Benchmark Endpoint (Functional Requirement 4.5):
    Executes search across:
    1. Hybrid Pipeline (Word2Vec Primary + FastText Fallback)
    2. Pure Word2Vec
    3. Pure FastText
    4. TF-IDF Lexical Baseline
    Returns side-by-side rankings to evaluate semantic superiority and OOV resilience.
    """
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search Engine is warming up. Please retry in a few seconds.",
        )
    try:
        return engine.compare_models(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running model comparison: {str(e)}",
        )


@router.get("/document/{document_id}", response_model=DocumentDetail, summary="Get Full Document")
def get_document(
    document_id: str,
    engine: SearchEngine = Depends(get_search_engine),
):
    """Fetches full content, tags, and category for a single knowledge base document."""
    if not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search Engine is warming up.",
        )

    doc = engine.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found.",
        )
    return doc
