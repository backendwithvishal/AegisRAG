import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState) -> dict:
    """
    Performs 2-stage retrieval:
      Stage 1: Bi-encoder cosine vector search in Qdrant (top 15 candidates)
      Stage 2: Cross-encoder FlashRank reranking (top 5 high-precision passages)
    """
    query = state.get("current_query", "")

    with logfire.span("🔍 Knowledge Retrieval", query=query[:80]):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")

        doc_contents = [doc["content"] for doc in raw_results if doc.get("content")]

        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info(f"Reranking complete. Kept top {len(reranked_contents)} most relevant chunks.")

        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]

    return {
        "documents": formatted_docs,
        "status": "Found technical context.",
        "plan": state.get("plan", []) + ["Context Retrieved"]
    }
