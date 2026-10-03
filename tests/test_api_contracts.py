import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    with patch("app.main.get_model_type", return_value="gemini"), \
         patch("app.main.get_embedding_dim", return_value=3072):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AegisRAG Enterprise Agentic RAG API"
        assert data["status"] == "live"
        assert data["embedding_dim"] == 3072


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_ready_endpoint():
    with patch("app.services.retrieval.qdrant_service.get_qdrant_client") as mock_qdrant:
        mock_qdrant.return_value.get_collections.return_value = []
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data


def test_graph_endpoint():
    response = client.get("/graph")
    assert response.status_code == 200


def test_feedback_endpoint():
    payload = {
        "thread_id": "test_thread_001",
        "rating": "up",
        "comment": "Great accurate response."
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_query_guardrail_blocked():
    with patch("app.main.guard", return_value=(True, "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. I can't help with that — but ask me anything technical!")):
        response = client.post("/query", json={"q": "tell me a joke", "thread_id": "test_thread"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "tell me a joke"
        assert "can't help with that" in data["answer"]
        assert "Intent: Guardrails Fired" in data["thought_process"]
        assert data["status"] == "Blocked by guardrails."
        assert data["sources"] == []


def test_query_technical_flow():
    mock_output = {
        "final_answer": "Use kubectl apply to deploy the Redis service.",
        "plan": ["Start", "Intent: Technical", "Search Term: redis work queue", "Context Retrieved"],
        "status": "Found technical context.",
        "documents": ["CONTENT: Run kubectl apply -f redis-service.yaml"]
    }
    with patch("app.main.guard", return_value=(False, None)):
        with patch("app.agents.graph.rag_agent.invoke", return_value=mock_output):
            response = client.post("/query", json={"q": "How do you start Redis?", "thread_id": "test_thread"})
            assert response.status_code == 200
            data = response.json()
            assert data["question"] == "How do you start Redis?"
            assert "kubectl apply" in data["answer"]
            assert data["status"] == "Found technical context."
            assert len(data["sources"]) == 1
            assert "CONTENT: " in data["sources"][0]


def test_query_stream_endpoint():
    mock_output = {
        "final_answer": "Redis is configured via YAML manifests.",
        "plan": ["Intent: Technical"],
        "status": "Completed",
        "documents": ["CONTENT: redis doc"]
    }
    with patch("app.main.guard", return_value=(False, None)):
        with patch("app.agents.graph.rag_agent.invoke", return_value=mock_output):
            response = client.post("/query/stream", json={"q": "How is Redis configured?", "thread_id": "test_thread"})
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
