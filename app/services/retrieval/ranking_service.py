import os
import time
from typing import List, Optional
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker: Optional[Ranker] = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank cross-encoder engine lazily.
    FlashRank uses a local quantized ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Cross-Encoder Model (ms-marco-MiniLM-L-6-v2) on CPU...")
        try:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "flashrank")
            os.makedirs(cache_dir, exist_ok=True)
            _ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2", cache_dir=cache_dir)
        except Exception:
            _ranker = Ranker(model_name="ms-marco-MiniLM-L-6-v2")
    return _ranker


def rerank_documents(query: str, documents: List[str], top_n: int = 5) -> List[str]:
    """
    Refines retrieval results by re-scoring passages against the query semantically.

    Why FlashRank:
      Standard bi-encoder vector search (Cosine Similarity) is fast but suffers from representation loss.
      Cross-Encoder reranking computes full token-to-token attention between query and candidate,
      eliminating noise and distractors. FlashRank executes quantized ONNX on CPU with <100ms latency.

    Args:
        query: User search query.
        documents: Raw retrieved candidate text passages.
        top_n: Number of top reranked passages to retain.

    Returns:
        List of top_n reranked document strings.
    """
    if not documents:
        return []

    start_time = time.time()
    with logfire.span("⚖️ FlashRank Reranking", candidate_count=len(documents), top_n=top_n):
        try:
            ranker = _get_ranker()

            passages = [
                {"id": i, "text": doc}
                for i, doc in enumerate(documents)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            reranked_docs = [res["text"] for res in results[:top_n]]
            duration = time.time() - start_time
            top_score = results[0]["score"] if results else "N/A"

            logfire.info(
                f"✅ FlashRank complete in {duration:.3f}s. "
                f"Retained top {len(reranked_docs)}/{len(documents)} chunks. "
                f"Top semantic score: {top_score}"
            )
            return reranked_docs

        except Exception as e:
            duration = time.time() - start_time
            logfire.error(f"❌ FlashRank Reranking Failed after {duration:.3f}s: {e}. Falling back to Qdrant order.")
            # Fallback to the original Qdrant order to ensure fault tolerance
            return documents[:top_n]
