"""
Core Semantic Search Service Engine.
Implements:
- Word2Vec dense embedding retrieval (Primary Model)
- FastText sub-word n-gram retrieval (Fallback Model for OOV words)
- Hybrid Fallback pipeline (Functional Requirement 4.2 & 4.3)
- TF-IDF baseline keyword retrieval (Functional Requirement 4.5)
- Category filtering and Top-K ranking (Functional Requirement 4.7)
"""
import time
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import joblib
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity

from app.config import (
    WORD2VEC_MODEL_PATH,
    FASTTEXT_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    PROCESSED_DF_PATH,
    VECTOR_DIMENSION,
    DEFAULT_TOP_K,
)
from app.models.schemas import (
    SearchMode,
    TokenAttribution,
    PreprocessingDetails,
    SearchResultItem,
    SearchResponse,
    CompareResponse,
    DocumentDetail,
    SystemStats,
)
from app.services.preprocessor import preprocess_query

logger = logging.getLogger("nexora.search_service")
logging.basicConfig(level=logging.INFO)


class SearchEngine:
    """
    Production-grade Search Engine maintaining in-memory models,
    precomputed L2-normalized vector matrices, and document metadata.
    """

    _instance: Optional["SearchEngine"] = None

    def __init__(self):
        self.w2v_model: Optional[Word2Vec] = None
        self.ft_model: Optional[FastText] = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.metadata_df: Optional[pd.DataFrame] = None
        
        # Precomputed and normalized numpy vector matrices for ultra-fast BLAS cosine similarity
        self.w2v_matrix_norm: Optional[np.ndarray] = None
        self.ft_matrix_norm: Optional[np.ndarray] = None
        
        # Category indices mapping for instant category filtering
        self.category_indices: Dict[str, np.ndarray] = {}
        self.categories: List[str] = []
        self.total_docs: int = 0
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "SearchEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        """Loads all serialized models, precomputes normalized matrices, and builds lookup indexes."""
        if self.is_loaded:
            logger.info("SearchEngine already initialized.")
            return

        t_start = time.time()
        logger.info("=== Initializing Nexora AI Search Engine ===")

        # 1. Load Word2Vec model
        logger.info(f"Loading Word2Vec model from {WORD2VEC_MODEL_PATH.name}...")
        self.w2v_model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))
        logger.info(f"-> Word2Vec loaded. Vocab size: {len(self.w2v_model.wv)}")

        # 2. Load FastText model
        logger.info(f"Loading FastText model from {FASTTEXT_MODEL_PATH.name}...")
        self.ft_model = FastText.load(str(FASTTEXT_MODEL_PATH))
        logger.info(f"-> FastText loaded. Vocab size: {len(self.ft_model.wv)}")

        # 3. Load TF-IDF assets
        logger.info(f"Loading TF-IDF vectorizer & matrix...")
        self.tfidf_vectorizer = joblib.load(str(TFIDF_VECTORIZER_PATH))
        self.tfidf_matrix = joblib.load(str(TFIDF_MATRIX_PATH))
        logger.info(f"-> TF-IDF loaded. Vocab size: {len(self.tfidf_vectorizer.get_feature_names_out())}, Matrix shape: {self.tfidf_matrix.shape}")

        # 4. Load Processed Documents DataFrame
        logger.info(f"Loading processed documents from {PROCESSED_DF_PATH.name}...")
        raw_df = pd.read_pickle(str(PROCESSED_DF_PATH))
        self.total_docs = len(raw_df)
        logger.info(f"-> Processed documents loaded: {self.total_docs} articles.")

        # 5. Extract and normalize vector matrices for high-throughput cosine similarity
        logger.info("Precomputing L2-normalized embedding matrices...")
        w2v_mat = np.vstack(raw_df["w2v_doc_vector"].values).astype(np.float32)
        w2v_norm_denom = np.linalg.norm(w2v_mat, axis=1, keepdims=True)
        w2v_norm_denom[w2v_norm_denom == 0] = 1e-9
        self.w2v_matrix_norm = w2v_mat / w2v_norm_denom

        ft_mat = np.vstack(raw_df["ft_doc_vector"].values).astype(np.float32)
        ft_norm_denom = np.linalg.norm(ft_mat, axis=1, keepdims=True)
        ft_norm_denom[ft_norm_denom == 0] = 1e-9
        self.ft_matrix_norm = ft_mat / ft_norm_denom

        # 6. Keep lightweight metadata dataframe for result formatting
        metadata_cols = [
            "document_id",
            "category",
            "title",
            "content",
            "keywords",
            "keywords_str",
            "cleaned_text_str",
        ]
        self.metadata_df = raw_df[metadata_cols].copy().reset_index(drop=True)

        # 7. Index categories for instant filtered queries
        self.categories = sorted(self.metadata_df["category"].dropna().unique().tolist())
        for cat in self.categories:
            self.category_indices[cat] = np.where(self.metadata_df["category"].values == cat)[0]

        self.is_loaded = True
        logger.info(f"=== Initialization Complete in {time.time() - t_start:.2f}s ===")

    def get_system_stats(self) -> SystemStats:
        """Returns metadata and health statistics about the loaded search models."""
        return SystemStats(
            status="ready" if self.is_loaded else "not_initialized",
            documents_count=self.total_docs,
            categories=self.categories,
            word2vec_vocab_size=len(self.w2v_model.wv) if self.w2v_model else 0,
            fasttext_vocab_size=len(self.ft_model.wv) if self.ft_model else 0,
            tfidf_vocab_size=len(self.tfidf_vectorizer.get_feature_names_out()) if self.tfidf_vectorizer else 0,
            vector_dimension=VECTOR_DIMENSION,
            model_paths={
                "word2vec": str(WORD2VEC_MODEL_PATH.name),
                "fasttext": str(FASTTEXT_MODEL_PATH.name),
                "tfidf": str(TFIDF_MATRIX_PATH.name),
                "dataset": str(PROCESSED_DF_PATH.name),
            },
        )

    def analyze_query_tokens(self, tokens: List[str]) -> Tuple[List[TokenAttribution], bool, List[str], List[str]]:
        """
        Inspects each query token to check if it exists in Word2Vec (Primary)
        or requires FastText sub-word n-gram fallback (OOV Handling).
        """
        attributions: List[TokenAttribution] = []
        fallback_tokens: List[str] = []
        primary_tokens: List[str] = []

        for token in tokens:
            if self.w2v_model and token in self.w2v_model.wv:
                attributions.append(
                    TokenAttribution(
                        token=token,
                        source="word2vec",
                        is_fallback=False,
                        details="Known term in Word2Vec vocabulary (Primary)",
                    )
                )
                primary_tokens.append(token)
            else:
                attributions.append(
                    TokenAttribution(
                        token=token,
                        source="fasttext",
                        is_fallback=True,
                        details="Out-of-vocabulary (OOV). FastText character n-grams took over (Fallback)",
                    )
                )
                fallback_tokens.append(token)

        fallback_triggered = len(fallback_tokens) > 0
        return attributions, fallback_triggered, fallback_tokens, primary_tokens

    def _compute_hybrid_query_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Builds the hybrid query embedding according to Functional Requirement 4.2 & 4.3:
        - Primary: Uses Word2Vec vector if token is in vocabulary.
        - Fallback: Uses FastText sub-word vector if token is unseen/OOV.
        Averages all vectors to form the final dense representation.
        """
        if not tokens:
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        token_vectors = []
        for token in tokens:
            if self.w2v_model and token in self.w2v_model.wv:
                token_vectors.append(self.w2v_model.wv[token])
            elif self.ft_model:
                token_vectors.append(self.ft_model.wv[token])

        if not token_vectors:
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        return np.mean(token_vectors, axis=0).astype(np.float32)

    def _compute_word2vec_query_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Builds query embedding strictly from Word2Vec.
        Unseen words are ignored or return zero vector if none match.
        """
        if not tokens or not self.w2v_model:
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        valid_vectors = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
        if not valid_vectors:
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        return np.mean(valid_vectors, axis=0).astype(np.float32)

    def _compute_fasttext_query_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Builds query embedding strictly using FastText sub-words for all tokens.
        """
        if not tokens or not self.ft_model:
            return np.zeros(VECTOR_DIMENSION, dtype=np.float32)

        vectors = [self.ft_model.wv[t] for t in tokens]
        return np.mean(vectors, axis=0).astype(np.float32)

    def _rank_dense_vectors(
        self,
        query_vector: np.ndarray,
        doc_matrix_norm: np.ndarray,
        top_k: int,
        category: Optional[str] = None,
        min_score: float = 0.0,
        model_name: str = "Hybrid",
    ) -> List[SearchResultItem]:
        """
        Computes cosine similarities via normalized BLAS dot product and extracts top-K ranked documents.
        """
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0 or doc_matrix_norm is None:
            return []

        normalized_q = (query_vector / q_norm).reshape(1, -1)
        # Cosine similarity: (N, D) @ (D, 1) -> (N,)
        similarities = np.dot(doc_matrix_norm, normalized_q.T).flatten()

        # Handle category filtering
        if category and category in self.category_indices:
            candidate_indices = self.category_indices[category]
            candidate_sims = similarities[candidate_indices]
        else:
            candidate_indices = np.arange(self.total_docs)
            candidate_sims = similarities

        # Filter by minimum score
        if min_score > 0.0:
            valid_mask = candidate_sims >= min_score
            candidate_indices = candidate_indices[valid_mask]
            candidate_sims = candidate_sims[valid_mask]

        if len(candidate_indices) == 0:
            return []

        # Top-K indices selection
        k = min(top_k, len(candidate_indices))
        if k < len(candidate_indices):
            # Fast partial sort
            top_part = np.argpartition(candidate_sims, -k)[-k:]
            sorted_within_top = top_part[np.argsort(-candidate_sims[top_part])]
        else:
            sorted_within_top = np.argsort(-candidate_sims)

        selected_doc_indices = candidate_indices[sorted_within_top]
        selected_scores = candidate_sims[sorted_within_top]

        # Build response items
        results: List[SearchResultItem] = []
        for rank, (doc_idx, score) in enumerate(zip(selected_doc_indices, selected_scores), start=1):
            row = self.metadata_df.iloc[doc_idx]
            raw_keywords = row["keywords"]
            if isinstance(raw_keywords, list):
                kw_list = raw_keywords[:8]
            elif isinstance(raw_keywords, str):
                kw_list = [k.strip() for k in raw_keywords.replace("[", "").replace("]", "").replace("'", "").split(",") if k.strip()][:8]
            else:
                kw_list = []

            # Create clean snippet
            content_str = str(row["content"])
            snippet = content_str[:220] + "..." if len(content_str) > 220 else content_str

            results.append(
                SearchResultItem(
                    rank=rank,
                    document_id=str(row["document_id"]),
                    category=str(row["category"]),
                    title=str(row["title"]),
                    content_snippet=snippet,
                    keywords=kw_list,
                    similarity_score=float(round(score, 4)),
                    model_used=model_name,
                )
            )

        return results

    def _search_tfidf(
        self,
        cleaned_text_str: str,
        top_k: int,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResultItem]:
        """
        Executes TF-IDF baseline keyword search using sparse matrix cosine similarity.
        """
        if not cleaned_text_str.strip() or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([cleaned_text_str])
        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        if category and category in self.category_indices:
            candidate_indices = self.category_indices[category]
            candidate_sims = similarities[candidate_indices]
        else:
            candidate_indices = np.arange(self.total_docs)
            candidate_sims = similarities

        if min_score > 0.0:
            valid_mask = candidate_sims >= min_score
            candidate_indices = candidate_indices[valid_mask]
            candidate_sims = candidate_sims[valid_mask]

        if len(candidate_indices) == 0:
            return []

        k = min(top_k, len(candidate_indices))
        if k < len(candidate_indices):
            top_part = np.argpartition(candidate_sims, -k)[-k:]
            sorted_within_top = top_part[np.argsort(-candidate_sims[top_part])]
        else:
            sorted_within_top = np.argsort(-candidate_sims)

        selected_doc_indices = candidate_indices[sorted_within_top]
        selected_scores = candidate_sims[sorted_within_top]

        results: List[SearchResultItem] = []
        for rank, (doc_idx, score) in enumerate(zip(selected_doc_indices, selected_scores), start=1):
            row = self.metadata_df.iloc[doc_idx]
            raw_keywords = row["keywords"]
            if isinstance(raw_keywords, list):
                kw_list = raw_keywords[:8]
            elif isinstance(raw_keywords, str):
                kw_list = [k.strip() for k in raw_keywords.replace("[", "").replace("]", "").replace("'", "").split(",") if k.strip()][:8]
            else:
                kw_list = []

            content_str = str(row["content"])
            snippet = content_str[:220] + "..." if len(content_str) > 220 else content_str

            results.append(
                SearchResultItem(
                    rank=rank,
                    document_id=str(row["document_id"]),
                    category=str(row["category"]),
                    title=str(row["title"]),
                    content_snippet=snippet,
                    keywords=kw_list,
                    similarity_score=float(round(score, 4)),
                    model_used="TF-IDF Baseline",
                )
            )

        return results

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> SearchResponse:
        """
        Main query interface executing semantic similarity search.
        """
        start_time = time.perf_counter()

        # 1. Preprocess query according to 4.1
        tokens, cleaned_text_str = preprocess_query(query)

        # 2. Analyze tokens for Word2Vec inclusion vs FastText fallback
        attributions, fallback_triggered, fallback_tokens, primary_tokens = self.analyze_query_tokens(tokens)

        prep_details = PreprocessingDetails(
            raw_query=query,
            cleaned_tokens=tokens,
            cleaned_text_str=cleaned_text_str,
            token_attributions=attributions,
            fallback_triggered=fallback_triggered,
            fallback_tokens=fallback_tokens,
            primary_tokens=primary_tokens,
        )

        results: List[SearchResultItem] = []

        if mode == SearchMode.HYBRID:
            # Primary space is Word2Vec document vectors, query uses Hybrid token vectors
            query_vec = self._compute_hybrid_query_vector(tokens)
            results = self._rank_dense_vectors(
                query_vec,
                self.w2v_matrix_norm,
                top_k=top_k,
                category=category,
                min_score=min_score,
                model_name="Hybrid (Word2Vec + FastText Fallback)",
            )

        elif mode == SearchMode.WORD2VEC:
            # Pure Word2Vec (fails on OOV tokens)
            query_vec = self._compute_word2vec_query_vector(tokens)
            results = self._rank_dense_vectors(
                query_vec,
                self.w2v_matrix_norm,
                top_k=top_k,
                category=category,
                min_score=min_score,
                model_name="Word2Vec (Primary)",
            )

        elif mode == SearchMode.FASTTEXT:
            # Pure FastText (uses FastText document space)
            query_vec = self._compute_fasttext_query_vector(tokens)
            results = self._rank_dense_vectors(
                query_vec,
                self.ft_matrix_norm,
                top_k=top_k,
                category=category,
                min_score=min_score,
                model_name="FastText (Sub-words)",
            )

        elif mode == SearchMode.TFIDF:
            # TF-IDF keyword baseline
            results = self._search_tfidf(
                cleaned_text_str,
                top_k=top_k,
                category=category,
                min_score=min_score,
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResponse(
            query=query,
            mode=mode.value,
            top_k=top_k,
            category_filter=category,
            total_corpus=self.total_docs,
            latency_ms=round(latency_ms, 2),
            preprocessing=prep_details,
            results=results,
        )

    def compare_models(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None,
    ) -> CompareResponse:
        """
        Executes query across all 4 configurations for side-by-side evaluation (Req 4.5).
        """
        start_time = time.perf_counter()

        tokens, cleaned_text_str = preprocess_query(query)
        attributions, fallback_triggered, fallback_tokens, primary_tokens = self.analyze_query_tokens(tokens)

        prep_details = PreprocessingDetails(
            raw_query=query,
            cleaned_tokens=tokens,
            cleaned_text_str=cleaned_text_str,
            token_attributions=attributions,
            fallback_triggered=fallback_triggered,
            fallback_tokens=fallback_tokens,
            primary_tokens=primary_tokens,
        )

        # 1. Hybrid Search
        hybrid_q = self._compute_hybrid_query_vector(tokens)
        hybrid_results = self._rank_dense_vectors(
            hybrid_q, self.w2v_matrix_norm, top_k=top_k, category=category, model_name="Hybrid"
        )

        # 2. Pure Word2Vec
        w2v_q = self._compute_word2vec_query_vector(tokens)
        w2v_results = self._rank_dense_vectors(
            w2v_q, self.w2v_matrix_norm, top_k=top_k, category=category, model_name="Word2Vec"
        )

        # 3. Pure FastText
        ft_q = self._compute_fasttext_query_vector(tokens)
        ft_results = self._rank_dense_vectors(
            ft_q, self.ft_matrix_norm, top_k=top_k, category=category, model_name="FastText"
        )

        # 4. TF-IDF Baseline
        tfidf_results = self._search_tfidf(cleaned_text_str, top_k=top_k, category=category)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return CompareResponse(
            query=query,
            top_k=top_k,
            category_filter=category,
            latency_ms=round(latency_ms, 2),
            preprocessing=prep_details,
            hybrid_results=hybrid_results,
            word2vec_results=w2v_results,
            fasttext_results=ft_results,
            tfidf_results=tfidf_results,
        )

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentDetail]:
        """Fetches full content and metadata for a specific document ID."""
        if self.metadata_df is None:
            return None

        match = self.metadata_df[self.metadata_df["document_id"] == doc_id]
        if match.empty:
            return None

        row = match.iloc[0]
        raw_keywords = row["keywords"]
        if isinstance(raw_keywords, list):
            kw_list = raw_keywords
        elif isinstance(raw_keywords, str):
            kw_list = [k.strip() for k in raw_keywords.replace("[", "").replace("]", "").replace("'", "").split(",") if k.strip()]
        else:
            kw_list = []

        return DocumentDetail(
            document_id=str(row["document_id"]),
            category=str(row["category"]),
            title=str(row["title"]),
            content=str(row["content"]),
            keywords=kw_list,
            cleaned_text_str=str(row["cleaned_text_str"]),
        )


def get_search_engine() -> SearchEngine:
    """Dependency provider for FastAPI route endpoints."""
    return SearchEngine.get_instance()
