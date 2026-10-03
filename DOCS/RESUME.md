# AegisRAG — Resume & Portfolio Presentation

### Project Overview
**AegisRAG** is a production-grade, fault-tolerant Enterprise Agentic RAG platform engineered with LangGraph, Portkey Gateway, NeMo Guardrails, FlashRank cross-encoder reranking, and Qdrant Cloud to deliver sub-1.5s latency QA over massive enterprise technical documentation with strict grounding and zero-trust input safety.

### Tech Stack
`Python` | `FastAPI` | `LangGraph` | `LangChain` | `Portkey Gateway` | `Groq (Llama 3.3 70B & 3.1 8B)` | `NeMo Guardrails` | `FlashRank (ONNX)` | `Qdrant Cloud` | `Gemini Embeddings (3072-dim)` | `RAGAS 0.4.3` | `Pydantic Logfire` | `LangSmith` | `Docker`

---

### Resume Bullet Points (Ready for Production AI Engineer Resumes)

* **Architected Enterprise Agentic RAG Platform with Two-Gate Defense**: Built a high-throughput, fault-tolerant QA backend using **FastAPI**, **LangGraph**, and **Portkey Gateway**, integrating **NeMo Guardrails** as a Gate-1 safety filter to reject off-topic/adversarial queries in <300ms, completely bypassing expensive 70B LLM token consumption.
* **Engineered 2-Stage CPU Retrieval Subsystem with 85% Noise Elimination**: Developed a hybrid vector retrieval pipeline combining **Qdrant Cloud** (3072-dim cosine search) and local quantized **FlashRank** cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Context Precision from 0.62 to 0.89 while sustaining <85ms local CPU execution latency.
* **Implemented High-Availability Gateway Routing & Semantic Caching**: Configured a resilient Portkey LLM gateway with automated exponential backoff retries (HTTP 429/503), caching, and seamless fallback from **Groq Llama 3.3 70B** to **Llama 3.1 8B Instant**, reducing p95 query latency to 2.1s and shielding against cloud provider outages.
* **Designed Full-Coverage Observability & 6-Metric RAGAS Evaluation Framework**: Instrumented distributed tracing with **Pydantic Logfire** and **LangSmith**, establishing an automated evaluation suite testing Faithfulness (0.92), Answer Relevancy (0.94), Context Precision (0.89), and custom Jaccard Tool Correctness (1.00) across 15 golden technical scenarios and 6 adversarial test cases.
* **Built Zero-Cost Universal Document Parsing & Incremental Ingestion**: Engineered an incremental parsing engine with SHA-256 content hashing supporting PDF, HTML, DOCX, and PPTX with paragraph-aware chunking, achieving zero external OCR costs and idempotent vector synchronization.

---

### Key Architectural Highlights for Interview Discussions
1. **Gate 1 vs Gate 2 Separation**: Why safety checks must happen before retrieval and reasoning to protect LLM token budgets.
2. **Bi-Encoder + Cross-Encoder Synergy**: How 2-stage retrieval balances scale (vector index over thousands of passages) with precision (cross-encoder query-passage cross-attention).
3. **Logfire Import Lifecycle**: Why `logfire.configure()` must precede all application imports to avoid silent tracer poisoning.
4. **Groq TPM Sliding-Window Management**: How batch size limits, context window truncation (300 chars / 2 chunks for evaluation judges), and cooldown buffers prevent 6,000 TPM rate-limiting errors.
