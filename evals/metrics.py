"""
Phase 2 - RAGAS 0.2.x + Tool Correctness metrics.

Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.
All LLM-based metrics run sample-by-sample with cooldowns between chunks
calibrated for Groq 6,000 TPM on_demand tier. Contexts are truncated to
300 chars (2 chunks max) so no single request exceeds the Groq token limit.

Compatibility notes (ragas 0.2.x)
----------------------------------
* ragas.metrics.collections does NOT exist in 0.2.x.
  Metrics live at ragas.metrics as singleton instances (lowercase names).
* langchain_community.chat_models.vertexai was removed from langchain-community.
  We patch it with a shim before ragas loads so the import never crashes.
* Scoring API: metric.single_turn_ascore(SingleTurnSample(...)) replaces
  the old abatch_score(inputs_list) call from earlier ragas versions.
"""

from __future__ import annotations

import os
import sys
import types
import asyncio
import logfire
import pandas as pd

JUDGE_MODEL = "llama-3.1-8b-instant"
COOLDOWN_STANDARD = 62        # seconds between experiments (Groq TPM window)
COOLDOWN_MINI = 40            # seconds between per-sample batches
GENERAL_BATCH_SIZE = 1        # one sample at a time
CONTEXT_TRUNCATE = 300        # chars per context chunk
CONTEXT_LIMIT = 2             # max context chunks per sample


# ---------------------------------------------------------------------------
# Compatibility shim -- patch the removed vertexai module BEFORE ragas loads.
# ---------------------------------------------------------------------------

def _patch_vertexai_shim() -> None:
    """
    ragas 0.2.x unconditionally imports langchain_community.chat_models.vertexai
    which was removed from langchain-community.  We inject a stub module so the
    import silently succeeds without needing the real google-cloud-aiplatform SDK.
    """
    key = "langchain_community.chat_models.vertexai"
    if key in sys.modules:
        return

    shim = types.ModuleType(key)

    class ChatVertexAI:  # noqa: D101 - stub only
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[key] = shim

    # Also attach as attribute on the parent package so attribute-style
    # access (langchain_community.chat_models.vertexai) also works.
    try:
        import langchain_community.chat_models as _parent  # noqa: PLC0415
        if not hasattr(_parent, "vertexai"):
            _parent.vertexai = shim
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_judge():
    """
    Returns (ragas_llm_wrapper, ragas_embeddings) ready to attach to metrics.
    Uses ChatGroq (JUDGE_GROQ key) so we never exhaust the production key.
    """
    _patch_vertexai_shim()
    from ragas.llms import LangchainLLMWrapper          # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    from langchain_groq import ChatGroq                  # noqa: PLC0415

    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY") or "dummy_judge_key"
    groq_llm = ChatGroq(model=JUDGE_MODEL, api_key=api_key, temperature=0.0)
    judge_llm = LangchainLLMWrapper(groq_llm)

    # ragas.embeddings.HuggingfaceEmbeddings is missing aembed_query / aembed_documents
    # (the two async abstract methods on BaseRagasEmbeddings), so it cannot be
    # instantiated directly.  LangchainEmbeddingsWrapper implements all 4 abstract
    # methods, delegating async calls to sync ones via a thread-pool executor.
    lc_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    return judge_llm, ragas_embeddings


async def _cooldown(seconds: int, label: str, status_cb=None) -> None:
    msg = f"Waiting {seconds}s after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb("Ready -- starting next experiment.")


def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks.
    """
    valid = []
    for s in golden_dataset.get("rag_samples", []):
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_ctx = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_ctx[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


async def _score_samples(
    metric,
    make_sample_fn,
    samples: list,
    status_cb=None,
    label: str = "",
) -> list:
    """
    Scores all samples one-by-one using metric.single_turn_ascore().
    Injects COOLDOWN_MINI between GENERAL_BATCH_SIZE chunks to stay under
    Groq's 6,000 TPM on_demand rate limit.
    """
    _patch_vertexai_shim()
    from ragas import SingleTurnSample  # noqa: PLC0415, F401 - imported for type usage

    all_scores: list = []
    batches = [
        samples[i: i + GENERAL_BATCH_SIZE]
        for i in range(0, len(samples), GENERAL_BATCH_SIZE)
    ]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        tasks = [metric.single_turn_ascore(make_sample_fn(s)) for s in batch]
        scores = await asyncio.gather(*tasks)
        all_scores.extend(scores)
    return all_scores


def _score_df(metric_key: str, samples: list, scores: list) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(sc), 3)}
        for s, sc in zip(samples, scores)
    ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 evaluation experiments.
    Returns a dict keyed by metric name mapping to a per-sample DataFrame.
    """
    _patch_vertexai_shim()

    # ragas 0.2.x: singletons live at ragas.metrics (NOT ragas.metrics.collections)
    from ragas.metrics import (  # noqa: PLC0415
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )

    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results: dict = {}

    with logfire.span("Eval Phase 2 -- All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"Exp 1/6 -- Faithfulness ({len(samples)} samples)...")
        with logfire.span("Exp 1 -- Faithfulness"):
            faithfulness.llm = judge_llm

            def _faith_sample(s):
                from ragas import SingleTurnSample  # noqa: PLC0415
                return SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                    retrieved_contexts=s["actual_contexts"],
                )

            scores = await _score_samples(
                faithfulness, _faith_sample, samples, status_cb, "Faithfulness"
            )
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"Exp 2/6 -- Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("Exp 2 -- Answer Relevancy"):
            answer_relevancy.llm = judge_llm
            answer_relevancy.embeddings = ragas_embeddings

            def _relevancy_sample(s):
                from ragas import SingleTurnSample  # noqa: PLC0415
                return SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                )

            scores = await _score_samples(
                answer_relevancy, _relevancy_sample, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"Exp 3/6 -- Context Precision ({len(samples)} samples)...")
        with logfire.span("Exp 3 -- Context Precision"):
            context_precision.llm = judge_llm

            def _ctx_prec_sample(s):
                from ragas import SingleTurnSample  # noqa: PLC0415
                return SingleTurnSample(
                    user_input=s["question"],
                    reference=s.get("reference", ""),
                    retrieved_contexts=s["actual_contexts"],
                )

            scores = await _score_samples(
                context_precision, _ctx_prec_sample, samples, status_cb, "Context Precision"
            )
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"Exp 4/6 -- Context Recall ({len(samples)} samples)...")
        with logfire.span("Exp 4 -- Context Recall"):
            context_recall.llm = judge_llm

            def _ctx_recall_sample(s):
                from ragas import SingleTurnSample  # noqa: PLC0415
                return SingleTurnSample(
                    user_input=s["question"],
                    reference=s.get("reference", ""),
                    retrieved_contexts=s["actual_contexts"],
                )

            scores = await _score_samples(
                context_recall, _ctx_recall_sample, samples, status_cb, "Context Recall"
            )
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness ─────────────────────────────────────────
        if status_cb:
            status_cb(f"Exp 5/6 -- Answer Correctness ({len(samples)} samples)...")
        with logfire.span("Exp 5 -- Answer Correctness"):
            answer_correctness.llm = judge_llm
            answer_correctness.embeddings = ragas_embeddings

            def _correctness_sample(s):
                from ragas import SingleTurnSample  # noqa: PLC0415
                return SingleTurnSample(
                    user_input=s["question"],
                    response=s["actual_response"],
                    reference=s.get("reference", ""),
                )

            scores = await _score_samples(
                answer_correctness, _correctness_sample, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", samples, scores)
            results["answer_correctness"] = df
            logfire.info("Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Tool Correctness (Jaccard -- zero LLM calls) ───────────────
        if status_cb:
            status_cb("Exp 6/6 -- Tool Correctness (zero LLM calls)...")
        with logfire.span("Exp 6 -- Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({
                    "question": s["question"][:65],
                    "tool_correctness": round(score, 3),
                })
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("All 6 experiments complete!")

    return results
