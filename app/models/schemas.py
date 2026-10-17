"""
Pydantic Schemas for API Requests, Responses, and System Payloads.
"""
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class SearchMode(str, Enum):
    HYBRID = "hybrid"          # Word2Vec primary + FastText OOV fallback (Recommended)
    WORD2VEC = "word2vec"      # Pure Word2Vec
    FASTTEXT = "fasttext"      # Pure FastText (sub-words)
    TFIDF = "tfidf"            # Baseline keyword matching


class TokenAttribution(BaseModel):
    token: str
    source: Literal["word2vec", "fasttext", "none"]
    is_fallback: bool
    details: str


class PreprocessingDetails(BaseModel):
    raw_query: str
    cleaned_tokens: List[str]
    cleaned_text_str: str
    token_attributions: List[TokenAttribution]
    fallback_triggered: bool
    fallback_tokens: List[str]
    primary_tokens: List[str]


class SearchResultItem(BaseModel):
    rank: int
    document_id: str
    category: str
    title: str
    content_snippet: str
    keywords: List[str]
    similarity_score: float
    model_used: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Retrieval model strategy")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    category: Optional[str] = Field(default=None, description="Optional category filter")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")


class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    category_filter: Optional[str] = None
    total_corpus: int
    latency_ms: float
    preprocessing: PreprocessingDetails
    results: List[SearchResultItem]


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query for comparative evaluation")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results per model")
    category: Optional[str] = Field(default=None, description="Optional category filter")


class CompareResponse(BaseModel):
    query: str
    top_k: int
    category_filter: Optional[str] = None
    latency_ms: float
    preprocessing: PreprocessingDetails
    hybrid_results: List[SearchResultItem]
    word2vec_results: List[SearchResultItem]
    fasttext_results: List[SearchResultItem]
    tfidf_results: List[SearchResultItem]


class DocumentDetail(BaseModel):
    document_id: str
    category: str
    title: str
    content: str
    keywords: List[str]
    cleaned_text_str: str


class SystemStats(BaseModel):
    status: str
    documents_count: int
    categories: List[str]
    word2vec_vocab_size: int
    fasttext_vocab_size: int
    tfidf_vocab_size: int
    vector_dimension: int
    model_paths: dict
