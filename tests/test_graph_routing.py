import pytest
from app.agents.graph import route_planner
from app.agents.state import AgentState


def test_route_planner_conversational():
    state: AgentState = {
        "messages": [{"role": "user", "content": "hello"}],
        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing...",
        "final_answer": ""
    }
    next_node = route_planner(state)
    assert next_node == "responder"


def test_route_planner_technical():
    state: AgentState = {
        "messages": [{"role": "user", "content": "How do I configure HPA?"}],
        "current_query": "Kubernetes HPA autoscaling configuration",
        "documents": [],
        "plan": ["Start", "Intent: Technical"],
        "status": "Searching...",
        "final_answer": ""
    }
    next_node = route_planner(state)
    assert next_node == "retriever"
