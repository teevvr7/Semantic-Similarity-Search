"""
Configuration and Environment Settings for Semantic Similarity Search.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"

# Model and Data Asset Paths (Modular layout with root fallback)
RAW_DATA_PATH = DATA_DIR / "raw" / "knowledge_base_improved 1.csv"
PROCESSED_DF_PATH = (
    DATA_DIR / "processed" / "processed_documents_df.pkl"
    if (DATA_DIR / "processed" / "processed_documents_df.pkl").exists()
    else BASE_DIR / "processed_documents_df.pkl"
)

WORD2VEC_MODEL_PATH = (
    MODELS_DIR / "word2vec_model.model"
    if (MODELS_DIR / "word2vec_model.model").exists()
    else BASE_DIR / "word2vec_model.model"
)
FASTTEXT_MODEL_PATH = (
    MODELS_DIR / "fasttext_model.model"
    if (MODELS_DIR / "fasttext_model.model").exists()
    else BASE_DIR / "fasttext_model.model"
)
TFIDF_VECTORIZER_PATH = (
    MODELS_DIR / "tfidf_vectorizer.pkl"
    if (MODELS_DIR / "tfidf_vectorizer.pkl").exists()
    else BASE_DIR / "tfidf_vectorizer.pkl"
)
TFIDF_MATRIX_PATH = (
    MODELS_DIR / "tfidf_matrix.pkl"
    if (MODELS_DIR / "tfidf_matrix.pkl").exists()
    else BASE_DIR / "tfidf_matrix.pkl"
)

# Service Settings
APP_TITLE = "Nexora AI | Semantic Similarity Search Engine"
APP_DESCRIPTION = (
    "Production-grade semantic similarity search system featuring a Primary Word2Vec "
    "dense retrieval model with FastText character n-gram OOV fallback and TF-IDF baseline."
)
APP_VERSION = "1.0.0"

# Search Defaults
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
VECTOR_DIMENSION = 100
HOST = "0.0.0.0"
PORT = 8000
