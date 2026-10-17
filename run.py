"""
Convenient Root Application Launcher for Nexora AI Semantic Similarity Search.
Usage:
    python run.py
"""
import sys
import time
import webbrowser
import threading
import uvicorn

BANNER = """
========================================================================
   _  __                     ___    ____
  / |/ /__ __ _____  _______/   |  /  _/
 /    / -_) \ / _  \/ __/ _  /| | _/ /  
/_/|_/\__/_\_\___/\_/_/  /_/ |_|/___/   
  Language Systems Group • Dense Semantic Retrieval Console
========================================================================
[*] Role: NLP Engineer Product Assessment (Ref: NX-NLP-2026-017)
[*] Primary Model: Word2Vec (100D CBOW/Skip-Gram Dense Embeddings)
[*] Fallback Model: FastText (Character n-grams 3-6 for unseen/OOV words)
[*] Baseline Model: TF-IDF Sparse Vector Space
[*] Document Corpus: 50,000 Indexed Knowledge Base Articles
[*] Server URL:     http://localhost:8000
[*] Swagger Docs:   http://localhost:8000/docs
========================================================================
"""


def open_browser():
    """Waits for model preloading to begin and opens the browser."""
    time.sleep(2.0)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    print(BANNER)
    print("Starting server... Preloading embeddings into memory for sub-10ms queries.\n")

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run(
            "app.main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[!] Server stopped cleanly. Goodbye!")
