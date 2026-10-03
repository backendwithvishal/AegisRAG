import pytest
from unittest.mock import patch
from app.services.retrieval.ranking_service import rerank_documents


def test_rerank_empty_documents():
    assert rerank_documents("query", []) == []


def test_rerank_fallback_on_exception():
    candidates = [
        "Candidate 1: Kubernetes Work Queues",
        "Candidate 2: Intel Xeon Processors",
        "Candidate 3: BGP Routing protocols"
    ]

    # Mock _get_ranker to raise an exception, testing the fault-tolerant fallback
    with patch("app.services.retrieval.ranking_service._get_ranker", side_effect=RuntimeError("ONNX Runtime Error")):
        result = rerank_documents("Kubernetes Work Queues", candidates, top_n=2)
        assert len(result) == 2
        # Must return the original candidate order gracefully without crashing
        assert result == candidates[:2]
