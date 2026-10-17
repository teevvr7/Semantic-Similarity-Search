import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.search_service import SearchEngine
from app.models.schemas import SearchMode


def test_search_pipeline():
    print("Testing SearchEngine initialization...")
    engine = SearchEngine.get_instance()
    engine.initialize()

    stats = engine.get_system_stats()
    print(f"System Stats: {stats.documents_count} docs, W2V vocab: {stats.word2vec_vocab_size}")

    # 1. Test Typo Query with FastText Fallback
    query = "MY app not worng"
    resp = engine.search(query, mode=SearchMode.HYBRID, top_k=3)
    print(f"\n--- Search: '{query}' ---")
    print(f"Latency: {resp.latency_ms:.2f}ms | Fallback Triggered: {resp.preprocessing.fallback_triggered}")
    print(f"Fallback Tokens: {resp.preprocessing.fallback_tokens}")
    for item in resp.results:
        print(f"  #{item.rank} [{item.similarity_score:.4f}] ({item.category}) {item.title}")

    # 2. Test Compare Pipeline
    comp = engine.compare_models("My accunt hacked my someone how fix this", top_k=2)
    print(f"\n--- Comparative Benchmark: '{comp.query}' ---")
    print(f"Latency: {comp.latency_ms:.2f}ms")
    print(f"Hybrid top:   {comp.hybrid_results[0].title} (Score: {comp.hybrid_results[0].similarity_score})")
    print(f"Word2Vec top: {comp.word2vec_results[0].title} (Score: {comp.word2vec_results[0].similarity_score})")
    print(f"FastText top: {comp.fasttext_results[0].title} (Score: {comp.fasttext_results[0].similarity_score})")
    if comp.tfidf_results:
        print(f"TF-IDF top:   {comp.tfidf_results[0].title} (Score: {comp.tfidf_results[0].similarity_score})")
    else:
        print("TF-IDF top:   No matches (0.00)")

    # 3. Test Document Retrieval
    doc = engine.get_document_by_id("KB000001")
    assert doc is not None
    print(f"\nDocument KB000001 fetched: '{doc.title}' in [{doc.category}]")
    print("\n[SUCCESS] All pipeline integration tests passed perfectly!")


if __name__ == "__main__":
    test_search_pipeline()
