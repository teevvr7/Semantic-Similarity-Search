# Nexora AI — Semantic Similarity Search System

[![FastAPI](https://img.shields.io/badge/FastAPI-0.121+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gensim](https://img.shields.io/badge/Gensim-4.4.0-E95420.svg?logo=python&logoColor=white)](https://radimrehurek.com/gensim/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Assessment-Confidential-blue.svg)](#)

A production-grade, modular Semantic Similarity Search System built for the **Nexora AI Language Systems Group** technical assessment (Assessment Ref: `NX-NLP-2026-017`).

The system ingests user natural language queries and retrieves the most semantically relevant documents across a knowledge base of **50,000 documents**. It features a **Word2Vec (100D)** primary dense retrieval model, an automatic **FastText character n-gram (3–6 chars)** fallback engine for unseen/out-of-vocabulary (OOV) words, and a **TF-IDF baseline** for comparative evaluation.

---

## 1. System Architecture & Workflow

```
+---------------------------------------------------------------------------------+
|                                USER INTERFACE                                   |
|   (Enterprise Web Console: Search Input, Token Attribution, Side-by-Side View)  |
+---------------------------------------------------------------------------------+
                                      |
                                      | REST API (POST /api/search, POST /api/compare)
                                      v
+---------------------------------------------------------------------------------+
|                            FASTAPI BACKEND SERVICE                              |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                       1. TEXT PREPROCESSING (Req 4.1)                           |
|      Lowercasing -> Regex Punctuation Strip -> NLTK Tokenize -> Lemmatization    |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                  2. TOKEN ATTRIBUTION & VECTOR ENCODING                         |
|                                                                                 |
|   Query Tokens: [ "app", "worng" ]                                              |
|                     |        |                                                  |
|       In Word2Vec?  YES      NO (OOV / Typo)                                    |
|                     |        |                                                  |
|                     v        v                                                  |
|             [Word2Vec 100D]  [FastText Sub-word n-grams 3-6D] (Req 4.3 Fallback)|
|                     \        /                                                  |
|                      v      v                                                   |
|             Combined Hybrid Mean Query Vector: q in R^100                       |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                 3. L2-NORMALIZED COSINE SIMILARITY (Req 4.4)                    |
|       Vectorized Matrix Dot Product: S = D_norm x q_norm^T                      |
|       Corpus: 50,000 Documents x 100 Dimensions -> Sub-10ms Latency            |
+---------------------------------------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------+
|                   4. CATEGORY FILTERING & RANKING (Req 4.7)                     |
|           Filter by Category (if chosen) -> Top-K Partial Sort -> Response       |
+---------------------------------------------------------------------------------+
```

---

## 2. Key Features

- **Primary Word2Vec Model (Req 4.2)**: 100-dimensional dense semantic vectors capturing conceptual relationships between words beyond exact string matching.
- **FastText OOV Fallback Engine (Req 4.3)**: When user queries contain unseen words, slang, or typographical errors (e.g., `"accunt"` or `"worng"`), the system seamlessly falls back to FastText character n-grams ($n \in [3, 6]$) to construct sub-word representations rather than dropping tokens or returning zero vectors.
- **Real-Time Token Explainability Ribbon**: Visualizes exactly how each query token was handled:
  - <span style="color:#059669; font-weight:bold;">Primary (Word2Vec)</span>: Term recognized in training vocabulary.
  - <span style="color:#d97706; font-weight:bold;">Fallback (FastText)</span>: Unseen / OOV term resolved via sub-words.
- **Side-by-Side Model Comparison Benchmark (Req 4.5)**: Evaluates any query across four strategies simultaneously:
  1. *Hybrid Pipeline (Word2Vec + FastText Fallback)*
  2. *Pure Word2Vec (fails on OOV terms)*
  3. *Pure FastText (all sub-word representation)*
  4. *TF-IDF Baseline (lexical keyword frequency)*
- **High-Throughput BLAS Cosine Search**: The 50,000 document embeddings are pre-stacked and $L_2$-normalized in memory upon server startup. Query similarity calculation reduces to an instantaneous BLAS matrix dot product with query execution latencies under **10 ms**.
- **Faceted Category Filtering & Top-K Controls (Req 4.7)**: Supports dynamic filtering across 15 document categories (*Authentication, Account Management, Payments, Shipping, Security, Mobile App, etc.*) with configurable result counts.
- **Enterprise Web UI**: Designed with clean typography, dark/light theme persistence, interactive slide-over document drawer, and pre-canned evaluation test cases.

---

## 3. Directory & Codebase Structure

The project strictly follows modular software engineering standards:

```
Semantic-Similarity-Search/
├── app/
│   ├── __init__.py                 # Application package marker
│   ├── config.py                   # Central configuration, asset paths & defaults
│   ├── main.py                     # FastAPI application factory, lifespan & static mount
│   ├── models/
│   │   ├── __init__.py             # Models package exports
│   │   └── schemas.py              # Pydantic schemas (SearchRequest, Responses, DTOs)
│   ├── services/
│   │   ├── __init__.py             # Services package exports
│   │   ├── preprocessor.py         # Standardized NLTK preprocessing & lemmatization
│   │   └── search_service.py       # Core singleton search engine, hybrid logic & BLAS ranking
│   ├── api/
│   │   ├── __init__.py             # API package exports
│   │   └── endpoints.py            # REST endpoints (/api/search, /api/compare, etc.)
│   └── static/
│       ├── index.html              # Modern, accessible enterprise search client
│       ├── css/
│       │   └── style.css           # High-contrast, tailored enterprise styling
│       └── js/
│           └── app.js              # State management, debouncing & UI orchestration
├── data/
│   ├── raw/
│   │   └── knowledge_base_improved 1.csv   # Raw knowledge base dataset
│   └── processed/
│       └── processed_documents_df.pkl      # Precomputed document embeddings & metadata (50k docs)
├── models/
│   ├── word2vec_model.model        # Serialized Word2Vec primary model
│   ├── fasttext_model.model        # Serialized FastText fallback model
│   ├── fasttext_model.model.wv...  # FastText sub-word n-gram vector weights (800 MB)
│   ├── tfidf_vectorizer.pkl        # Serialized TF-IDF vectorizer
│   └── tfidf_matrix.pkl            # Serialized TF-IDF document matrix
├── notebooks/
│   ├── data-understanding.ipynb    # Exploratory data understanding notebook
│   └── Semantic_Similarity_Search(Project5).ipynb # Core model training & validation notebook
├── docs/
│   └── NLP_P5_Technical_Assessment_Semantic_Search.pdf # Assessment design brief
├── tests/
│   └── test_pipeline.py            # End-to-end integration test suite
├── run.py                          # Root launcher script with ASCII banner & auto-browser
├── requirements.txt                # Production dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Comprehensive documentation
```

---

## 4. NLP Methodology & Functional Requirements

### 4.1 Text Preprocessing (`app/services/preprocessor.py`)
Conforming to **Requirement 4.1**, queries and documents undergo an identical deterministic pipeline:
1. **Case Normalization**: Conversion of all characters to lowercase.
2. **Noise Removal**: Regex filtering `[^a-z\s]` stripping punctuation, digits, and control characters.
3. **Tokenization**: NLTK `word_tokenize` splits strings into individual lexical tokens.
4. **Stopword Elimination**: English stopwords (`nltk.corpus.stopwords`) are stripped.
5. **Length Filtering**: Single-character tokens (`len(token) <= 1`) are removed.
6. **Morphological Lemmatization**: WordNet Lemmatizer (`WordNetLemmatizer().lemmatize()`) reduces words to base lemmas.

### 4.2 Text Representation (`app/services/search_service.py`)
Dense semantic vectors are generated by computing the mean vector across token embeddings:
$$\vec{v}_{\text{doc}} = \frac{1}{|T|} \sum_{t \in T} \vec{w}_t \quad \in \mathbb{R}^{100}$$

### 4.3 Unseen Words & Hybrid Fallback Engine
When a query token $t$ is submitted:
- **Case 1 ($t \in \mathcal{V}_{\text{W2V}}$)**: Retrieve vector directly from the primary Word2Vec vocabulary:
  $$\vec{e}_t = \mathbf{W}_{\text{w2v}}(t)$$
- **Case 2 ($t \notin \mathcal{V}_{\text{W2V}}$)**: The fallback mechanism triggers. FastText segments $t$ into character n-grams $g \in \mathcal{G}_t$ ($3 \le |g| \le 6$) and sums their embeddings:
  $$\vec{e}_t = \sum_{g \in \mathcal{G}_t} \mathbf{W}_{\text{ft}}(g)$$

This guarantees that misspelled queries (e.g. `"MY app not worng"`) or emerging terminology still map to meaningful semantic representations rather than degrading to zero vectors.

### 4.4 Similarity Metric & Vector Optimization
Cosine similarity measures the angular distance between the query vector $\vec{q}$ and document vectors $\vec{d}_i$:
$$\text{CosineSimilarity}(\vec{q}, \vec{d}_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\|_2 \|\vec{d}_i\|_2}$$

By precomputing unit vectors $\hat{d}_i = \frac{\vec{d}_i}{\|\vec{d}_i\|_2}$ at server startup:
$$\vec{S} = \mathbf{D}_{\text{norm}} \cdot \hat{q}^T$$
This single vectorized matrix multiplication searches all 50,000 documents in **under 10 milliseconds**.

### 4.5 Baseline Comparison (TF-IDF)
The app provides an interactive side-by-side benchmark with TF-IDF. This empirically demonstrates why dense semantic representations outperform keyword matching when queries contain paraphrases, synonyms, or typos.

---

## 5. How to Activate and Run the Web App

### Step 1: Clone or Navigate to the Workspace Directory
Ensure you are in the project root directory:
```bash
cd "Semantic-Similarity-Search"
```

### Step 2: Create and Activate a Virtual Environment
It is strongly recommended to use an isolated Python virtual environment (`venv`):

**On Windows (PowerShell):**
```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1
```
> *Tip: If script execution is disabled in PowerShell, enable it for the current session via:*  
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

**On Windows (Command Prompt - cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip & Install Python Dependencies
With the virtual environment activated, install all required packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application
Launch the web server using the dedicated runner:
```bash
python run.py
```

*Alternatively, launch via Uvicorn directly:*
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> **Note on Initial Startup:**  
> On startup, the server preloads the 800 MB FastText model weights and 50,000 document embeddings into RAM. This takes approximately **25–45 seconds**. Once loaded, all subsequent queries execute in **sub-10ms** real-time.

### Step 5: Access the Web App & Documentation
- **Interactive Web App**: [http://localhost:8000](http://localhost:8000)
- **FastAPI Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 6. Verification & Test Suite

An end-to-end integration test is provided in `tests/test_pipeline.py`. It tests model loading, hybrid OOV fallback, comparative evaluation, and document retrieval.

Run the test suite with:
```bash
python tests/test_pipeline.py
```

Expected output:
```text
System Stats: 50000 docs, W2V vocab: 697

--- Search: 'MY app not worng' ---
Latency: 7.2ms | Fallback Triggered: True
Fallback Tokens: ['worng']
  #1 [0.5203] (Technical Support) A Guide to Setting Up Fresh Install the Easy Way
  #2 [0.5185] (Technical Support) Set Up App Reinstall Step by Step

--- Comparative Benchmark: 'My accunt hacked my someone how fix this' ---
Hybrid top:   Fix Closed Ticket Access Problems on iOS (Score: 0.6765)
FastText top: Fix Compromised Account Access Problems in Minutes (Score: 0.7483)

[SUCCESS] All pipeline integration tests passed perfectly!
```

---

## 7. REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check & model initialization state |
| `GET` | `/api/stats` | Indexed documents count, vocabularies, and vector dimensions |
| `GET` | `/api/categories` | Array of all 15 document categories for faceted filters |
| `POST` | `/api/search` | Execute semantic search (`HYBRID`, `WORD2VEC`, `FASTTEXT`, `TFIDF`) |
| `POST` | `/api/compare` | Execute query across all 4 retrieval strategies side-by-side |
| `GET` | `/api/document/{id}` | Retrieve full article body, tags, and metadata by document ID |

---

## 8. Nexora AI Assessment Summary Table

| Requirement Ref | Requirement Description | Implementation in Web App |
|---|---|---|
| **4.1** | Text Preprocessing | `app/services/preprocessor.py` (lowercasing, regex cleaning, tokenization, lemmatization) |
| **4.2** | Text Representation | 100D dense vectors via Word2Vec Skip-Gram / CBOW |
| **4.3** | Unseen Words (OOV) | FastText sub-word n-gram fallback + real-time UI token attribution |
| **4.4** | Cosine Similarity & Ranking | Precomputed $L_2$-normalized BLAS matrix dot product + Top-K ranking |
| **4.5** | Baseline Comparison | Side-by-side benchmark comparing Hybrid, Word2Vec, FastText, and TF-IDF |
| **4.6** | Generalization | Dynamic preprocessing and encoding handles arbitrary unseen input |
| **4.7** | Result Count & Filters | User-configurable Top-K (3 to 25) and 15-category filtering |
