import math
import re
from typing import List, Dict, Any, Optional
import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_query, get_embedding_dim


_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
    return _client


def _simple_bm25_score(query_tokens: List[str], doc_tokens: List[str], avg_dl: float = 200.0, k1: float = 1.5, b: float = 0.75) -> float:
    """Zero-cost local BM25 scoring helper for hybrid retrieval ranking."""
    score = 0.0
    doc_len = len(doc_tokens)
    for q in query_tokens:
        if not q:
            continue
        tf = doc_tokens.count(q)
        if tf > 0:
            idf = 1.0  # Normalized local term importance
            denom = tf + k1 * (1 - b + b * (doc_len / avg_dl))
            score += idf * (tf * (k1 + 1)) / denom
    return score


def search_enterprise_knowledge(
    query: str,
    limit: int = 15,
    source_type: Optional[str] = None,
    hybrid: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Performs high-precision retrieval over enterprise knowledge.
    Uses modern Qdrant query_points interface with 3072-dim or 768-dim embeddings.

    Args:
        query: User or planner generated search query.
        limit: Number of candidates to retrieve (default: 15 for 2-stage reranking).
        source_type: Optional filter for document origin ("true", "noisy", etc.).
        hybrid: Override flag for enabling hybrid BM25 + dense vector fusion.

    Returns:
        List of dicts: [{"content": str, "source": str, "source_type": str, "score": float}]
    """
    client = get_qdrant_client()
    dim = get_embedding_dim()
    collection_name = settings.get_collection_name(dim)
    enable_hybrid = settings.ENABLE_HYBRID_SEARCH if hybrid is None else hybrid

    with logfire.span("🔍 Vector Retrieval (Qdrant)", query=query[:80], collection=collection_name, limit=limit):
        try:
            # Check if collection exists
            if not client.collection_exists(collection_name):
                logfire.warning(f"Collection '{collection_name}' not found in Qdrant.")
                return []

            query_vector = embed_query(query)

            # Build metadata filter if source_type is specified
            query_filter = None
            if source_type:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value=source_type)
                        )
                    ]
                )

            # Modern query_points interface
            fetch_limit = limit * 2 if enable_hybrid else limit
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=fetch_limit,
                query_filter=query_filter,
                with_payload=True
            )

            results: List[Dict[str, Any]] = []
            for res in response.points:
                payload = res.payload or {}
                results.append({
                    "content": payload.get("text", ""),
                    "source": payload.get("source", "Unknown"),
                    "source_type": payload.get("source_type", "general"),
                    "score": float(res.score) if res.score is not None else 0.0
                })

            # Hybrid Reciprocal Rank Fusion (RRF) / BM25 re-weighting if active
            if enable_hybrid and results:
                q_tokens = re.findall(r"\w+", query.lower())
                for item in results:
                    d_tokens = re.findall(r"\w+", item["content"].lower())
                    bm25_val = _simple_bm25_score(q_tokens, d_tokens)
                    # Hybrid combined score: 70% dense cosine + 30% lexical BM25
                    item["hybrid_score"] = item["score"] * 0.7 + (min(bm25_val, 5.0) / 5.0) * 0.3

                results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                results = results[:limit]

            logfire.info(f"✅ Qdrant returned {len(results)} candidate points.")
            return results

        except Exception as e:
            logfire.error(f"❌ Qdrant Search Failed: {e}")
            return []
