# Changelog

All notable changes to the **AegisRAG** project are documented in this file using [Conventional Commits](https://www.conventionalcommits.org/).

## [1.0.0] - 2026-10-04

### Architecture & Core
- **feat(agent)**: implement LangGraph state machine with Planner, Retriever, and Responder nodes.
- **feat(guardrails)**: introduce NeMo Guardrails Gate 1 with Colang v1.0 rules and deterministic indicator matching.
- **feat(gateway)**: integrate Portkey AI Gateway with fallback routing (`@rag/llama-3.3-70b` -> `@brag/llama-3.1-8b`), retry strategies, and cache status extraction.
- **feat(retrieval)**: build 2-stage retrieval pipeline using Qdrant vector search and FlashRank (`ms-marco-MiniLM-L-6-v2`) cross-encoder reranking on CPU.

### Ingestion & Processing
- **feat(ingestion)**: implement universal document ingestion processor supporting PDF, HTML, DOCX, PPTX, and TXT with paragraph boundary chunking.
- **feat(ingestion)**: add incremental ingestion using SHA-256 content hashing to avoid redundant re-indexing.
- **fix(retrieval)**: isolate Qdrant collections by embedding dimension (`enterprise_rag_3072` vs `enterprise_rag_768`) to eliminate vector dimension clashes.

### API & DevOps
- **feat(api)**: provide FastAPI service with `/query`, `/query/stream` (SSE), `/feedback` (SQLite), `/health`, `/ready`, and `/graph` endpoints.
- **feat(security)**: add output guardrail and PII redaction for leaked keys, tokens, emails, and credentials.
- **feat(evals)**: create automated RAGAS 0.4.3 multi-metric judge suite with AsyncOpenAI judge, 40s TPM cooldowns, and markdown reporting CLI.
- **feat(benchmarks)**: add end-to-end latency benchmarks (p50/p95/p99) and architectural ablation study suite.
- **ci(github-actions)**: add automated CI workflow with linting, unit tests, and indicator regression tests.
