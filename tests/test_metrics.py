import pandas as pd
import pytest
from evals.metrics import _prep_samples, _score_df, _safe_mean


def test_prep_samples_filters_empty_responses():
    dataset = {
        "rag_samples": [
            {
                "question": "Q1",
                "actual_response": "Some response",
                "relevant_contexts": ["Context 1", "Context 2"],
            },
            {
                "question": "Q2",
                "actual_response": "   ",
                "relevant_contexts": ["Context 1"],
            },
            {
                "question": "Q3",
                "actual_response": "",
            },
        ]
    }
    valid = _prep_samples(dataset)
    assert len(valid) == 1
    assert valid[0]["question"] == "Q1"
    assert len(valid[0]["actual_contexts"]) == 2


def test_prep_samples_fallback_context():
    dataset = {
        "rag_samples": [
            {
                "question": "Q1",
                "actual_response": "Valid answer",
                "actual_contexts": [],
                "relevant_contexts": [],
            }
        ]
    }
    valid = _prep_samples(dataset)
    assert len(valid) == 1
    assert valid[0]["actual_contexts"] == ["No context available."]


def test_score_df_safe_conversions():
    samples = [
        {"question": "What is Redis?"},
        {"question": "How does Raft work?"},
        {"question": "Missing score"},
    ]
    scores = [0.8523, None, Exception("API Timeout")]
    df = _score_df("faithfulness", samples, scores)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df.loc[0, "faithfulness"] == 0.852
    assert df.loc[1, "faithfulness"] == 0.0
    assert df.loc[2, "faithfulness"] == 0.0


def test_safe_mean():
    df = pd.DataFrame({"score": [0.8, 0.9, 0.7]})
    assert _safe_mean(df, "score") == 0.8

    empty_df = pd.DataFrame(columns=["score"])
    assert _safe_mean(empty_df, "score") == 0.0

    nan_df = pd.DataFrame({"score": [float("nan"), float("nan")]})
    assert _safe_mean(nan_df, "score") == 0.0
