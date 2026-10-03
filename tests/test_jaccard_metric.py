import pytest
from evals.pipeline import detect_tool


def test_detect_tool_technical():
    thought_process = ["Start", "Intent: Technical", "Search Term: redis work queue", "Context Retrieved"]
    tool = detect_tool(thought_process)
    assert tool == "retrieve_documents"


def test_detect_tool_conversational():
    thought_process = ["Start", "Intent: Conversational/Memory", "Retrieval: Skipped"]
    tool = detect_tool(thought_process)
    assert tool == "direct_answer"


def test_detect_tool_guardrails():
    thought_process = ["Intent: Guardrails Fired", "Retrieval: Skipped"]
    tool = detect_tool(thought_process)
    assert tool == "guardrails"


def test_jaccard_metric_calculation():
    def jaccard(actual: list, expected: list) -> float:
        s_act = set(actual)
        s_exp = set(expected)
        union = len(s_act | s_exp)
        return len(s_act & s_exp) / union if union > 0 else 0.0

    assert jaccard(["retrieve_documents"], ["retrieve_documents"]) == 1.0
    assert jaccard(["direct_answer"], ["retrieve_documents"]) == 0.0
    assert jaccard(["retrieve_documents", "other"], ["retrieve_documents"]) == 0.5
