"""
====================================================================================================
ENTERPRISE AGENTIC RAG SYSTEM - COMPREHENSIVE PROJECT CONTEXT & TECHNICAL MASTER SPECIFICATION
====================================================================================================

This file serves as the definitive, all-inclusive architectural and operational blueprint 
for the Enterprise Agentic Retrieval-Augmented Generation (RAG) system. Every subsystem, 
engineering pattern, design choice, mathematical formulation, safety constraint, and 
evaluation methodology is documented herein without omission.

TABLE OF CONTENTS:
------------------
1. System Overview, Vision, and Core Philosophies
2. Complete Tech Stack & Component Matrix
3. High-Level Architectural Flow & Lifecycles
4. Ingestion Engine & Local Multi-Format Document Parsing
5. Vector Storage & Two-Stage Semantic Retrieval (Qdrant + FlashRank)
6. Safety & Input Filtering Gate (NVIDIA NeMo Guardrails & Colang)
7. LLM Gateway & Resiliency Proxy (Portkey AI)
8. Agentic Brain & Cyclic State Machine (LangGraph Core)
9. Observability, Distributed Tracing & Logging (Logfire + LangSmith)
10. Architectural Gotchas & Production Engineering Decisions
11. Comprehensive Evaluation Suite (RAGAS, Guardrails, & Tool Correctness)
12. Environment Variables & Configuration Reference
13. Complete Module, Class, & Function Manifest
14. API Contracts, Schemas, & Payloads
====================================================================================================
"""

import os
from typing import Dict, List, Any


# ==================================================================================================
# 1. SYSTEM OVERVIEW, VISION, AND CORE PHILOSOPHIES
# ==================================================================================================
SYSTEM_OVERVIEW: Dict[str, Any] = {
    "title": "Enterprise Agentic RAG (Scalable Pipeline)",
    "mission": (
        "Deliver a production-grade, fault-tolerant, high-throughput question-answering system "
        "capable of ingesting complex enterprise documentation (Kubernetes, Intel hardware, networking) "
        "and answering both conversational queries and deep technical questions with zero hallucinations, "
        "guaranteed grounding, strict input safety, and automated quantitative evaluation."
    ),
    "core_philosophies": [
        {
            "name": "Query Intent Segregation",
            "description": (
                "Treating every query identically leads to either high latency or poor accuracy. The "
                "system segregates requests into Conversational (resolved via thread memory) and "
                "Technical (routed to semantic vector search and multi-stage reranking)."
            )
        },
        {
            "name": "True Data vs. Noisy Data Distinction",
            "description": (
                "In enterprise environments, knowledge bases contain authoritative technical documents "
                "(True Data) alongside irrelevant or distractor files (Noisy Data). The pipeline uses "
                "two-stage Cross-Encoder semantic reranking to discard noise and isolate high-signal chunks."
            )
        },
        {
            "name": "Defense in Depth (Gatekeeper Safety)",
            "description": (
                "NeMo Guardrails acts as Gate 1 prior to expensive retrieval and LLM generation. "
                "Adversarial jailbreaks, off-topic chit-chat, and out-of-scope prompts are stopped "
                "at the gate in milliseconds without consuming vector search or 70B model tokens."
            )
        },
        {
            "name": "Local, Zero-Cost Cloud-Agnostic Utilities",
            "description": (
                "Document parsing (PDF, HTML, DOCX, PPTX, TXT) and semantic cross-encoder reranking "
                "(FlashRank ONNX TinyBERT) run 100% locally on CPU, ensuring zero per-query reranking "
                "costs, enhanced data privacy, and absence of external OCR/API lock-in."
            )
        },
        {
            "name": "Full Observability & Rigorous Offline Evaluation",
            "description": (
                "Every internal step is traced with Pydantic Logfire and LangSmith. System performance "
                "is quantitatively governed by a 6-metric RAGAS eval suite running against a verified "
                "15-sample Golden Dataset and 6 Guardrail benchmark test cases."
            )
        }
    ]
}


# ==================================================================================================
# 2. COMPLETE TECH STACK & COMPONENT MATRIX
# ==================================================================================================
TECH_STACK_MATRIX: Dict[str, Dict[str, str]] = {
    "Orchestration & Workflow": {
        "Framework": "LangChain & LangGraph (StateGraph)",
        "State Management": "LangGraph MemorySaver (thread_id persistence)",
        "Execution": "Cyclic state machine with conditional routing"
    },
    "Large Language Models": {
        "Primary Synthesis Engine": "Groq Llama 3.3 70B Versatile (via Portkey)",
        "Fallback Engine": "Groq Llama 3.1 8B Instant (via Portkey)",
        "Guardrails Classification": "Groq Llama 3.1 8B Instant (via ChatGroq)",
        "Evaluation Judge": "Groq Llama 3.1 8B Instant (via AsyncOpenAI / JUDGE_GROQ)"
    },
    "LLM Gateway & Proxy": {
        "Provider": "Portkey AI Gateway (PORTKEY_GATEWAY_URL)",
        "Capabilities": "Fallback routing, simple/semantic caching, exponential retry, trace logging"
    },
    "Safety & Policy Enforcement": {
        "Engine": "NVIDIA NeMo Guardrails",
        "Policy Definition Language": "Colang (v1.0 syntax) + YAML configuration",
        "Detection Strategy": "Deterministic RAIL_INDICATORS substring verification"
    },
    "Vector Database": {
        "Provider": "Qdrant Cloud (Managed Cluster)",
        "Distance Metric": "Cosine Similarity (models.Distance.COSINE)",
        "Collection Name": "enterprise_rag",
        "API Method": "query_points (modern Qdrant client standard)"
    },
    "Embeddings Layer": {
        "Primary Model": "Google Gemini (gemini-embedding-2-preview, 3072 dimensions)",
        "Fallback Model": "SentenceTransformers (all-mpnet-base-v2, 768 dimensions)",
        "RAGAS Eval Embeddings": "HuggingFace (sentence-transformers/all-MiniLM-L6-v2)"
    },
    "Semantic Reranking": {
        "Engine": "FlashRank (Local CPU Cross-Encoder)",
        "Model": "ms-marco-MiniLM-L-6-v2 (Quantized ONNX format)",
        "Latency": "< 100ms for 15 candidate passages"
    },
    "Local Document Ingestion": {
        "PDF": "pypdf (primary) + pdfplumber (image-heavy page fallback)",
        "HTML": "BeautifulSoup4 (tag decomposition & clean text extraction)",
        "Office Documents": "python-docx (.docx) + python-pptx (.pptx) / unstructured partition",
        "Plain Text": "Built-in UTF-8 file readers (.txt, .md)"
    },
    "API & Backend": {
        "Framework": "FastAPI (ASGI)",
        "Server": "Uvicorn",
        "Data Validation": "Pydantic v2 (BaseModel, Settings)"
    },
    "Observability & Monitoring": {
        "System Tracing": "Pydantic Logfire (distributed spans, latency, errors)",
        "Agent Tracing": "LangSmith (v2 tracing, graph transitions, prompts, token counts)"
    },
    "Evaluation Suite": {
        "Framework": "RAGAS 0.4.3 (Collections metrics + AsyncOpenAI judge)",
        "Tool Metric": "Custom Jaccard Set Overlap (Zero LLM cost)",
        "Guardrail Metric": "Binary Confusion Matrix (TP, TN, FP, FN, Precision, Recall, Accuracy)"
    }
}


# ==================================================================================================
# 3. HIGH-LEVEL ARCHITECTURAL FLOW & LIFECYCLES
# ==================================================================================================
ARCHITECTURAL_LIFECYCLES: Dict[str, str] = {
    "query_execution_lifecycle": """
    ========================================================================================
    STEP-BY-STEP QUERY EXECUTION FLOW:
    ========================================================================================
    1. Client Request:
       - Client sends POST request to FastAPI endpoint /query with body {"q": str, "thread_id": str}.
       
    2. Gate 1: NeMo Guardrails Check:
       - FastAPI calls guard(q) using NeMo Guardrails powered by Llama 3.1 8B.
       - NeMo classifies the user intent against Colang rules (off-topic, jailbreak, dialog).
       - If a rail fires (e.g. user asks for a joke or attempts jailbreak):
         * guard() detects one of the distinctive RAIL_INDICATORS.
         * FastAPI immediately halts execution and returns the canned bot refusal.
         * LangGraph, Qdrant, FlashRank, and Llama 3.3 70B are NEVER invoked.
         
    3. Gate 2: LangGraph State Machine Initialization:
       - If guardrails pass, initial AgentState is constructed with:
         * messages: [{"role": "user", "content": q}]
         * current_query: q
         * documents: []
         * plan: ["Start"]
         * status: "Initializing Graph..."
       - rag_agent.invoke(initial_state, config={"configurable": {"thread_id": thread_id}}) is called.
       - MemorySaver recalls historical conversation turns associated with thread_id.
       
    4. Planner Node Execution:
       - Portkey-backed LLM (ChatOpenAI) analyzes conversation history + latest message.
       - Decides whether the query is CONVERSATIONAL or TECHNICAL.
       - If CONVERSATIONAL (greeting, meta-question, memory recall):
         * Sets current_query = 'CONVERSATIONAL'
         * Sets plan = ["Intent: Conversational/Memory", "Retrieval: Skipped"]
         * Conditional edge routes directly to Responder Node (bypassing search).
       - If TECHNICAL (Kubernetes, Intel, Networking):
         * Synthesizes a focused, standalone search term.
         * Sets current_query = <search_term>
         * Sets plan = ["Intent: Technical", "Search Term: <search_term>"]
         * Conditional edge routes to Retriever Node.
         
    5. Retriever Node Execution:
       - Calls search_enterprise_knowledge(query, limit=15) via Qdrant Cloud.
       - Generates 3072-dim query embedding using Gemini (gemini-embedding-2-preview).
       - Executes Cosine Similarity vector search against Qdrant collection 'enterprise_rag'.
       - Fetches top 15 candidate text chunks.
       - Passes 15 candidates into rerank_documents(query, candidates, top_n=5) via FlashRank.
       - FlashRank (local ONNX Cross-Encoder) evaluates query-passage pairs simultaneously.
       - Retains top 5 highest-scoring relevant chunks, prepends 'CONTENT: ', updates state.
       
    6. Responder Node Execution:
       - Formats prompt: includes technical context (if technical) and conversation history.
       - Invokes Portkey native client (portkey_client.chat.completions.create).
       - Portkey applies fallback strategy (@rag/llama-3.3-70b -> @brag/llama-3.1-8b) and caching.
       - Responder inspects response headers for x-portkey-cache-status.
       - If Cache HIT: updates plan with 'Cache: Hit ⚡'.
       - Appends final assistant message to history and stores in MemorySaver.
       
    7. Response Dispatch:
       - FastAPI serializes response:
         {
           "question": q,
           "answer": <final_answer>,
           "thought_process": <plan_steps>,
           "status": <status_message>,
           "sources": <retrieved_chunks>
         }
    ========================================================================================
    """,

    "ingestion_lifecycle": """
    ========================================================================================
    STEP-BY-STEP DATA INGESTION PIPELINE:
    ========================================================================================
    1. Directory Traversal:
       - run_universal_ingestion scans DATA/ directory and subdirectories (true_data, noisy_data).
       - Subdirectory name automatically determines metadata source_type ('true' vs 'noisy').
       
    2. Collection Lifecycle:
       - If --wipe flag is passed, existing Qdrant collection 'enterprise_rag' is dropped.
       - get_embedding_dim() probes embedding backend to establish vector dimension (3072).
       - Creates Qdrant collection with Cosine distance metric if non-existent.
       
    3. Document Parsing:
       - .pdf: Local parsing via pypdf; blank/scanned pages retry using pdfplumber.
       - .html / .htm: BeautifulSoup removes <script>, <style>, <meta>, extracts clean text.
       - .docx / .pptx: Extracted using python-docx / python-pptx / unstructured partition.
       - .txt / .md: UTF-8 plain text loader.
       
    4. Paragraph-Aware Semantic Chunking:
       - chunk_text() splits full text by double newlines ('\\n\\n').
       - Accumulates paragraphs into chunks up to 1500 characters without mid-sentence cuts.
       
    5. Local Staging & Persistence:
       - Saves chunk metadata and content locally to processed_data/<source_type>/<filename>.json.
       
    6. Vector Embedding & Qdrant Upsert:
       - Chunks are vectorized in batches of 50 via embed_texts().
       - PointStruct records are constructed with UUIDs, vectors, and payload:
         {"text": chunk, "source": filename, "source_type": source_type}.
       - Points are upserted into Qdrant Cloud collection.
    ========================================================================================
    """
}


# ==================================================================================================
# 4. INGESTION ENGINE & LOCAL MULTI-FORMAT DOCUMENT PARSING
# ==================================================================================================
INGESTION_ENGINE_SPEC: Dict[str, Any] = {
    "design_principles": [
        "Zero external OCR or cloud parsing costs — all extraction executes locally on CPU.",
        "Deterministic separation of True (gold technical) and Noisy (distractor) documents.",
        "Local JSON staging in processed_data/ to maintain an auditable snapshot of chunk boundaries."
    ],
    "parsers": {
        "pdf_loader": {
            "file": "app/ingestion/loaders/pdf.py",
            "function": "parse_pdf(file_path: str) -> str",
            "strategy": (
                "Opens PDF with pypdf.PdfReader. Iterates through all pages extracting text. "
                "Tracks blank pages. If any pages returned empty text, falls back to pdfplumber "
                "for those specific pages. Handles large multi-page PDFs seamlessly."
            )
        },
        "html_loader": {
            "file": "app/ingestion/loaders/html.py",
            "function": "parse_html(file_path: str) -> str",
            "strategy": (
                "Parses HTML with BeautifulSoup(content, 'html.parser'). Explicitly decomposes "
                "['script', 'style', 'meta', 'noscript'] tags. Extracts text with newline separators "
                "and collapses redundant whitespace and blank lines."
            )
        },
        "office_loader": {
            "file": "app/ingestion/loaders/office.py",
            "function": "parse_office(file_path: str) -> str",
            "strategy": (
                "Parses .docx and .pptx files using unstructured partition (or python-docx / python-pptx). "
                "Extracts paragraph elements and slide shape texts into clean text strings."
            )
        },
        "text_loader": {
            "file": "app/ingestion/loaders/text.py",
            "function": "parse_text(file_path: str) -> str",
            "strategy": "Opens plain text files with UTF-8 encoding and ignore error handling."
        }
    },
    "chunking_strategy": {
        "file": "app/ingestion/chunking/splitter.py",
        "function": "chunk_text(text: str, chunk_size: int = 1500) -> List[str]",
        "algorithm": (
            "Splits text on double newlines ('\\n\\n') to preserve paragraph integrity. "
            "Iterates through paragraphs, appending to current_chunk until length exceeds 1500 chars. "
            "When exceeded, pushes current_chunk to list and begins a new chunk. Never cuts off "
            "words or sentences mid-thought."
        )
    }
}


# ==================================================================================================
# 5. VECTOR STORAGE & TWO-STAGE SEMANTIC RETRIEVAL (QDRANT + FLASHRANK)
# ==================================================================================================
RETRIEVAL_SUBSYSTEM_SPEC: Dict[str, Any] = {
    "bi_encoder_stage": {
        "layer": "Stage 1: Fast Candidate Retrieval",
        "engine": "Qdrant Cloud Vector Database",
        "embedding_model": "gemini-embedding-2-preview (Google Generative AI)",
        "embedding_dimension": 3072,
        "distance_metric": "Cosine Similarity",
        "candidate_count": 15,
        "rationale": (
            "Bi-Encoders embed query and documents independently into vector space. Vector dot product / "
            "cosine distance across thousands of pre-computed embeddings executes in sub-10ms. "
            "However, bi-encoders cannot model fine-grained token-level cross-interactions between "
            "the query terms and passage content."
        ),
        "resiliency": (
            "Includes retry logic with exponential backoff (1s -> 2s -> 4s -> 8s) for Gemini API rate limits. "
            "Includes automatic probe and fallback to sentence-transformers/all-mpnet-base-v2 (768-dim) "
            "if Gemini key is unavailable or quota is exceeded."
        )
    },
    "cross_encoder_stage": {
        "layer": "Stage 2: Deep Semantic Reranking",
        "engine": "FlashRank (Local CPU Engine)",
        "model": "ms-marco-MiniLM-L-6-v2 (ONNX Quantized)",
        "final_top_k": 5,
        "rationale": (
            "Cross-Encoders take the query string and document text concatenated together and pass them "
            "through all transformer attention layers simultaneously. This captures deep linguistic nuance, "
            "negations, and exact technical contexts that bi-encoders miss. FlashRank executes this locally "
            "using highly optimized ONNX runtimes in <100ms on CPU with zero per-call cost."
        ),
        "custom_vs_langchain_wrapper": [
            "Granular Observability: Direct Logfire span instrumentation tracking exact ms latency and top scores.",
            "Lazy Loading: Prevents heavy ONNX model instantiation at startup, avoiding server boot delays.",
            "Bulletproof Fallback: Catches any memory/execution failure and automatically returns raw Qdrant order.",
            "Lightweight State: Eliminates heavy LangChain Document object overhead in favor of pure strings."
        ]
    }
}


# ==================================================================================================
# 6. SAFETY & INPUT FILTERING GATE (NVIDIA NEMO GUARDRAILS & COLANG)
# ==================================================================================================
GUARDRAILS_SPEC: Dict[str, Any] = {
    "architecture_role": "Gate 1 Safety Filter placed in app/main.py before LangGraph invocation.",
    "model": "Groq Llama 3.1 8B Instant (low latency, minimal token cost for intent matching)",
    "policy_rules": {
        "off_topic_rail": {
            "intent": "user ask off topic (jokes, weather, recipes, general math, movies)",
            "bot_refusal": (
                "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, and networking. "
                "I can't help with that — but ask me anything technical!"
            )
        },
        "jailbreak_rail": {
            "intent": "user attempt jailbreak ('ignore previous instructions', 'DAN mode', 'developer mode')",
            "bot_refusal": (
                "I maintain consistent guidelines regardless of how I am prompted. "
                "I am here to help with Kubernetes, Intel, and networking. What can I help you with?"
            )
        },
        "dialog_rails": {
            "greeting": "Returns standard Enterprise IT Assistant greeting.",
            "capabilities": "Lists domain coverage (Kubernetes, Intel Hardware, Enterprise Networking).",
            "farewell": "Returns polite sign-off."
        }
    },
    "the_rail_indicators_mechanism": {
        "problem": (
            "NeMo Guardrails generate() method returns a dictionary with 'content' as a plain string. "
            "It does NOT provide an explicit boolean 'fired=True' flag indicating whether a rail or flow triggered."
        ),
        "solution": (
            "The system defines RAIL_INDICATORS — a list of unique, verbatim substrings from the defined "
            "bot responses that would never appear in a legitimate technical answer about Kubernetes/Networking. "
            "The guard() function checks: any(ind in content for ind in RAIL_INDICATORS). "
            "If True -> request blocked, canned response returned. If False -> request passed to LangGraph."
        ),
        "indicators_list": [
            "can't help with that — but ask me anything technical",
            "I maintain consistent guidelines regardless of how I am prompted",
            "Hello! I'm your Enterprise IT Assistant",
            "Goodbye! Feel free to return whenever you have more enterprise IT questions",
            "I'm an Enterprise AI Assistant with deep expertise in"
        ]
    }
}


# ==================================================================================================
# 7. LLM GATEWAY & RESILIENCY PROXY (PORTKEY AI)
# ==================================================================================================
LLM_GATEWAY_SPEC: Dict[str, Any] = {
    "role": "Centralized reverse proxy routing all production LLM calls through Portkey.",
    "configuration": {
        "strategy": "fallback",
        "primary_target": "@rag/llama-3.3-70b-versatile (Groq)",
        "fallback_target": "@brag/llama-3.1-8b-instant (Groq)",
        "retry_policy": "2 attempts on HTTP 429 (rate limit) and 503 (service unavailable)",
        "caching": "simple mode (exact match response caching; semantic on Enterprise tier)",
        "request_timeout": "Hard timeout cap in milliseconds"
    },
    "dual_client_pattern": {
        "native_portkey_client": {
            "usage": "Used in Responder Node (app/agents/nodes/responder.py)",
            "purpose": (
                "Calls portkey_client.chat.completions.create. Allows inspecting underlying response "
                "headers (via extract_cache_status) for 'x-portkey-cache-status'. When 'HIT', the node "
                "records 'Cache: Hit ⚡' in the thought process without paying model latency."
            )
        },
        "langchain_chatopenai_wrapper": {
            "usage": "Used in Planner Node (app/agents/nodes/planner.py)",
            "purpose": (
                "Returns ChatOpenAI(base_url=PORTKEY_GATEWAY_URL, model='@rag/llama-3.3-70b-versatile'). "
                "Integrates seamlessly with LangGraph invoke() syntax while routing through Portkey headers."
            )
        }
    }
}


# ==================================================================================================
# 8. AGENTIC BRAIN & CYCLIC STATE MACHINE (LANGGRAPH CORE)
# ==================================================================================================
LANGGRAPH_CORE_SPEC: Dict[str, Any] = {
    "state_definition": {
        "file": "app/agents/state.py",
        "schema": """
        class AgentState(TypedDict):
            messages: Annotated[List[dict], operator.add]  # Appends chat history
            current_query: str                            # Active search term or 'CONVERSATIONAL'
            documents: List[str]                          # Top reranked technical context strings
            plan: List[str]                               # Step-by-step thought process trail
            status: str                                   # Current human-readable execution status
            final_answer: str                             # Synthesized output text
        """
    },
    "nodes": {
        "planner_node": {
            "file": "app/agents/nodes/planner.py",
            "logic": (
                "Examines the full conversation history and current user query. Determines if the question "
                "can be answered purely from conversation history / greeting ('CONVERSATIONAL') or if it "
                "requires fresh documentation search ('TECHNICAL'). For technical queries, outputs a refined "
                "search term optimized for vector retrieval."
            )
        },
        "retriever_node": {
            "file": "app/agents/nodes/retriever.py",
            "logic": (
                "Executes Qdrant vector retrieval (top 15) followed by FlashRank cross-encoder reranking (top 5). "
                "Formats chunks with 'CONTENT: ' prefixes and stores in state['documents']."
            )
        },
        "responder_node": {
            "file": "app/agents/nodes/responder.py",
            "logic": (
                "Constructs the final prompt. For conversational queries, grounds answers in conversation history. "
                "For technical queries, strictly bounds answers to the retrieved context chunks (up to 25k chars). "
                "Executes Portkey synthesis and detects cache hits."
            )
        }
    },
    "graph_topology": {
        "file": "app/agents/graph.py",
        "entry_point": "planner",
        "conditional_edge": "route_planner: if current_query == 'CONVERSATIONAL' -> responder, else -> retriever",
        "static_edges": ["retriever -> responder", "responder -> END"],
        "checkpointer": "MemorySaver() linked to thread_id"
    }
}


# ==================================================================================================
# 9. OBSERVABILITY, DISTRIBUTED TRACING & LOGGING (LOGFIRE + LANGSMITH)
# ==================================================================================================
OBSERVABILITY_SPEC: Dict[str, Any] = {
    "pydantic_logfire": {
        "role": "Distributed system infrastructure tracing and APM.",
        "spans_tracked": [
            "FastAPI /query lifecycle & latency",
            "NeMo Guardrails evaluation duration & verdicts",
            "Planner intent classification spans",
            "Qdrant vector query execution and point count",
            "FlashRank cross-encoder reranking latency & top scores",
            "Document parsing duration per loader (PDF, HTML, Office, Text)",
            "Vector embedding batching and API retries"
        ]
    },
    "langsmith": {
        "role": "Agentic LLM and graph state transition tracing.",
        "metrics_tracked": [
            "LangGraph state transitions between Planner, Retriever, and Responder",
            "Exact prompt templates, system instructions, and variable interpolations",
            "Token usage (prompt tokens, completion tokens, total cost)",
            "Thread history and MemorySaver checkpoint recall"
        ]
    }
}


# ==================================================================================================
# 10. ARCHITECTURAL GOTCHAS & PRODUCTION ENGINEERING DECISIONS
# ==================================================================================================
ENGINEERING_GOTCHAS: List[Dict[str, str]] = [
    {
        "gotcha": "Logfire Process Poisoning Bug",
        "problem": (
            "If any Python module invokes logfire.info() or logfire.span() before logfire.configure() "
            "has run in the process, Logfire enters a silent permanent no-op mode, discarding all subsequent spans."
        ),
        "solution": (
            "In app/main.py, load_dotenv() and logfire.configure(token=os.getenv('LOGFIRE_TOKEN')) are "
            "executed at lines 1-10 BEFORE importing any internal application packages (config, graph, rails)."
        )
    },
    {
        "gotcha": "ML Model & SDK Lazy Loading Pattern",
        "problem": (
            "Instantiating heavy models (FlashRank ONNX) or cloud SDKs (Gemini embeddings) at module level "
            "causes sluggish FastAPI startup (>10s) and risks early logfire poisoning."
        ),
        "solution": (
            "Wrap models in lazy initializers (_init(), _get_ranker()). Models load on the first incoming "
            "query, ensuring instant server startup and health check readiness."
        )
    },
    {
        "gotcha": "Groq On-Demand Rate Limits (6,000 TPM)",
        "problem": (
            "Evaluating multiple RAG samples concurrently with 70B or 8B models easily spikes burst token "
            "consumption above Groq's 6,000 TPM limit, triggering HTTP 429/413 errors."
        ),
        "solution": (
            "1. Two-Key Strategy: GROQ_API_KEY for live traffic; JUDGE_GROQ for eval workloads.\n"
            "2. GENERAL_BATCH_SIZE = 1: Process 1 sample at a time.\n"
            "3. Cooldowns: 40s between individual samples, 62s between full experiment suites.\n"
            "4. Context Truncation: Truncate context chunks to 300 chars (max 2 chunks) during judging."
        )
    },
    {
        "gotcha": "Unstructured Library Segfaults on Office Files",
        "problem": "Heavy unstructured dependencies can cause segmentation faults on native platforms.",
        "solution": "Direct usage of python-docx and python-pptx in evaluation data parser (evals/data_parser.py)."
    }
]


# ==================================================================================================
# 11. COMPREHENSIVE EVALUATION SUITE (RAGAS, GUARDRAILS, & TOOL CORRECTNESS)
# ==================================================================================================
EVALUATION_SUITE_SPEC: Dict[str, Any] = {
    "golden_dataset": {
        "rag_samples_count": 15,
        "guardrails_samples_count": 6,
        "sources": [
            "DATA/true_data/parallel_work_queue.txt (Kubernetes Redis Work Queues)",
            "DATA/true_data/pods_autoscale.html (HPA / VPA Autoscaling)",
            "DATA/true_data/job_management.html (Databricks Job Management API/CLI)",
            "DATA/true_data/cronjobs.docx (Kubernetes CronJobs Spec & Concurrency)",
            "DATA/true_data/monitor_job.docx (Kubernetes Job Monitoring & Conditions)"
        ]
    },
    "ragas_043_api_standards": [
        "Rule 1: Use llm_factory('llama-3.1-8b-instant', provider='openai', client=AsyncOpenAI(...)) — do NOT use LangchainLLMWrapper.",
        "Rule 2: Use AsyncOpenAI client targeting https://api.groq.com/openai/v1 — do NOT use sync client.",
        "Rule 3: Call await metric.abatch_score(list_of_dicts) — do NOT use evaluate() wrapper.",
        "Rule 4: Embeddings for relevancy/correctness use HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2', use_api=False)."
    ],
    "the_6_metrics": [
        {
            "id": 1,
            "metric": "Faithfulness",
            "library": "RAGAS",
            "type": "Generation / Grounding",
            "needs": ["user_input", "response", "retrieved_contexts"],
            "what_it_measures": "Measures whether every claim in the response is strictly supported by the retrieved context chunks (zero hallucinations)."
        },
        {
            "id": 2,
            "metric": "Answer Relevancy",
            "library": "RAGAS",
            "type": "Generation / Topic Alignment",
            "needs": ["user_input", "response"],
            "what_it_measures": "Measures how directly the response addresses the user prompt by generating synthetic questions and comparing cosine similarity."
        },
        {
            "id": 3,
            "metric": "Context Precision",
            "library": "RAGAS",
            "type": "Retrieval / Ranking Quality",
            "needs": ["user_input", "reference", "retrieved_contexts"],
            "what_it_measures": "Measures whether the most relevant, high-signal context chunks are ranked at the top of the context list by FlashRank."
        },
        {
            "id": 4,
            "metric": "Context Recall",
            "library": "RAGAS",
            "type": "Retrieval / Completeness",
            "needs": ["user_input", "reference", "retrieved_contexts"],
            "what_it_measures": "Measures whether the retriever fetched all necessary factual sentences required to construct the ground-truth reference answer."
        },
        {
            "id": 5,
            "metric": "Answer Correctness",
            "library": "RAGAS",
            "type": "End-to-End Quality",
            "needs": ["user_input", "response", "reference"],
            "what_it_measures": "Combines factual statement overlap (precision/recall) and semantic embedding similarity against the ground truth reference."
        },
        {
            "id": 6,
            "metric": "Tool Correctness",
            "library": "Custom / Jaccard",
            "type": "Agent Routing Quality",
            "needs": ["actual_tools_called", "expected_tools"],
            "formula": "Jaccard Index = |called ∩ expected| / |called ∪ expected|",
            "what_it_measures": "Deterministic check verifying that the Planner routed to retrieve_documents for technical queries and direct_answer for conversational queries. 100% free, 0 tokens."
        }
    ],
    "guardrails_metrics": {
        "categories": ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"],
        "formulas": {
            "Precision": "TP / (TP + FP)",
            "Recall": "TP / (TP + FN)",
            "Accuracy": "(TP + TN) / Total"
        }
    }
}


# ==================================================================================================
# 12. ENVIRONMENT VARIABLES & CONFIGURATION REFERENCE
# ==================================================================================================
ENV_VARIABLES_SPEC: Dict[str, Dict[str, str]] = {
    "GROQ_API_KEY": {
        "description": "Primary API key for Groq reasoning engine (Llama 3.3 70B & Llama 3.1 8B Guardrails).",
        "format": "gsk_..."
    },
    "GROQ_FALLBACK_API_KEY": {
        "description": "Secondary Groq API key configured in Portkey for automatic fallback routing.",
        "format": "gsk_..."
    },
    "PORTKEY_API_KEY": {
        "description": "Portkey AI Gateway API key enabling routing, caching, retries, and analytics.",
        "format": "pk-..."
    },
    "GEMINI_API_KEY": {
        "description": "Google Gemini API key for gemini-embedding-2-preview (3072-dim embeddings).",
        "format": "AIza..."
    },
    "QDRANT_API_KEY": {
        "description": "Qdrant Cloud cluster access token.",
        "format": "Secret string"
    },
    "QDRANT_CLUSTER_ENDPOINT": {
        "description": "Full HTTPS URL of the Qdrant Cloud cluster (including port :6333).",
        "format": "https://xyz-example.cloud.qdrant.io:6333"
    },
    "LOGFIRE_TOKEN": {
        "description": "Pydantic Logfire authentication token for distributed tracing.",
        "format": "logfire_..."
    },
    "LANGSMITH_TRACING": {
        "description": "Enables LangSmith tracing for LangChain / LangGraph ('true' / 'false').",
        "format": "true"
    },
    "LANGSMITH_API_KEY": {
        "description": "LangSmith personal / service token.",
        "format": "lsv2_..."
    },
    "LANGSMITH_PROJECT": {
        "description": "Project workspace name inside LangSmith dashboard.",
        "format": "rag_scale_test"
    },
    "LANGSMITH_ENDPOINT": {
        "description": "LangSmith ingestion API endpoint.",
        "format": "https://api.smith.langchain.com"
    },
    "JUDGE_GROQ": {
        "description": "Separate Groq API key dedicated exclusively to RAGAS LLM-as-judge scoring.",
        "format": "gsk_..."
    }
}


# ==================================================================================================
# 13. COMPLETE MODULE, CLASS, & FUNCTION MANIFEST
# ==================================================================================================
CODE_MANIFEST: Dict[str, Dict[str, Any]] = {
    "app/config.py": {
        "classes": ["Settings"],
        "attributes": [
            "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION",
            "GROQ_API_KEY", "GROQ_MODEL", "GROQ_FALLBACK_API_KEY", "PORTKEY_API_KEY",
            "GROQ_SLUG", "GROQ_SLUG_2", "LANGSMITH_TRACING", "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"
        ]
    },
    "app/main.py": {
        "endpoints": [
            "GET / - Health check",
            "GET /graph - Returns Mermaid diagram PNG bytes of LangGraph workflow",
            "POST /query - Primary query execution endpoint (Guardrails + LangGraph)"
        ],
        "models": ["QueryRequest(q: str, thread_id: Optional[str] = 'default_user')"]
    },
    "app/agents/state.py": {
        "classes": ["AgentState(TypedDict)"],
        "fields": ["messages", "current_query", "documents", "plan", "status", "final_answer"]
    },
    "app/agents/graph.py": {
        "functions": ["route_planner(state: AgentState) -> str"],
        "objects": ["workflow: StateGraph", "checkpointer: MemorySaver", "rag_agent: CompiledStateGraph"]
    },
    "app/agents/nodes/planner.py": {
        "functions": ["planner_node(state: AgentState) -> dict"]
    },
    "app/agents/nodes/retriever.py": {
        "functions": ["retrieve_node(state: AgentState) -> dict"]
    },
    "app/agents/nodes/responder.py": {
        "functions": ["generate_node(state: AgentState) -> dict"]
    },
    "app/gateway/client.py": {
        "constants": ["GATEWAY_CONFIG", "portkey_client"],
        "functions": [
            "get_langchain_llm(feature: str = 'rag') -> ChatOpenAI",
            "extract_cache_status(response) -> str"
        ]
    },
    "app/guardrails/rails.py": {
        "functions": [
            "initialize_rails() -> None",
            "guard(message: str) -> tuple[bool, str | None]"
        ]
    },
    "app/guardrails/colang_rules.py": {
        "constants": ["COLANG_CONTENT", "YAML_CONTENT", "RAIL_INDICATORS"]
    },
    "app/ingestion/processor.py": {
        "functions": [
            "save_processed_locally(data: dict, source_type: str, filename: str) -> str",
            "process_file(file_path: str, filename: str, source_type: str)",
            "process_directory(dir_path: str, source_type: str)",
            "run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False)"
        ]
    },
    "app/ingestion/chunking/splitter.py": {
        "functions": ["chunk_text(text: str, chunk_size: int = 1500) -> List[str]"]
    },
    "app/ingestion/loaders/pdf.py": {
        "functions": ["parse_pdf(file_path: str) -> str"]
    },
    "app/ingestion/loaders/html.py": {
        "functions": ["parse_html(file_path: str) -> str"]
    },
    "app/ingestion/loaders/office.py": {
        "functions": ["parse_office(file_path: str) -> str"]
    },
    "app/ingestion/loaders/text.py": {
        "functions": ["parse_text(file_path: str) -> str"]
    },
    "app/services/retrieval/embedding.py": {
        "functions": [
            "get_embedding_dim() -> int",
            "embed_query(query: str) -> list[float]",
            "embed_texts(texts: list[str]) -> list[list[float]]"
        ]
    },
    "app/services/retrieval/qdrant_service.py": {
        "functions": ["search_enterprise_knowledge(query: str, limit: int = 8) -> list[dict]"]
    },
    "app/services/retrieval/ranking_service.py": {
        "functions": [
            "_get_ranker() -> Ranker",
            "rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]"
        ]
    },
    "evals/pipeline.py": {
        "functions": [
            "detect_tool(thought_process: list) -> str",
            "run_pipeline(golden_dataset: dict, progress_callback=None) -> dict",
            "save_results(dataset: dict, path: str) -> None",
            "load_golden_dataset() -> dict"
        ]
    },
    "evals/guardrails_eval.py": {
        "functions": [
            "_is_blocked(response_json: dict) -> bool",
            "run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list",
            "compute_guardrails_metrics(results: list) -> dict"
        ]
    },
    "evals/metrics.py": {
        "functions": [
            "_build_judge() -> tuple",
            "_cooldown(seconds: int, label: str, status_cb=None)",
            "_prep_samples(golden_dataset: dict) -> list",
            "_score_df(metric_key: str, samples: list, scores) -> pd.DataFrame",
            "_batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = '') -> list",
            "run_all_metrics(golden_dataset: dict, status_cb=None) -> dict"
        ]
    },
    "evals/data_parser.py": {
        "functions": [
            "_parse_docx(file_path: str) -> str",
            "_parse_pptx(file_path: str) -> str",
            "parse_file(file_path: str) -> str",
            "load_all_chunks() -> list[dict]"
        ]
    }
}


# ==================================================================================================
# 14. API CONTRACTS, SCHEMAS, & PAYLOADS
# ==================================================================================================
API_SCHEMAS: Dict[str, Any] = {
    "POST /query request": {
        "headers": {"Content-Type": "application/json"},
        "payload": {
            "q": "How do you configure horizontal pod autoscaling for CPU in Kubernetes?",
            "thread_id": "session_user_uuid_12345"
        }
    },
    "POST /query response (Technical Success)": {
        "status_code": 200,
        "body": {
            "question": "How do you configure horizontal pod autoscaling for CPU in Kubernetes?",
            "answer": "To configure Horizontal Pod Autoscaling (HPA) based on CPU utilization...",
            "thought_process": [
                "Intent: Technical",
                "Search Term: horizontal pod autoscaler cpu configuration kubernetes",
                "Context Retrieved"
            ],
            "status": "Found technical context.",
            "sources": [
                "CONTENT: The Horizontal Pod Autoscaler automatically scales the number of Pods...",
                "CONTENT: Target average CPU utilization can be set via metrics.resource.target..."
            ]
        }
    },
    "POST /query response (Guardrails Blocked)": {
        "status_code": 200,
        "body": {
            "question": "Write a python script to brute force an SSH server password",
            "answer": "I maintain consistent guidelines regardless of how I am prompted. I am here to help with Kubernetes, Intel, and networking. What can I help you with?",
            "thought_process": [
                "Intent: Guardrails Fired",
                "Retrieval: Skipped"
            ],
            "status": "Blocked by guardrails.",
            "sources": []
        }
    },
    "POST /query response (Conversational / Memory)": {
        "status_code": 200,
        "body": {
            "question": "Hello there! What was the first question I asked you?",
            "answer": "Hello! Your first question was regarding Horizontal Pod Autoscaling in Kubernetes.",
            "thought_process": [
                "Intent: Conversational/Memory",
                "Retrieval: Skipped"
            ],
            "status": "Handling conversationally (using memory)...",
            "sources": []
        }
    }
}


# ==================================================================================================
# SUMMARY HELPER FOR DEVELOPERS / AGENTS
# ==================================================================================================
def print_project_summary() -> None:
    """Prints a quick human-readable summary of the Enterprise RAG system context."""
    print("=" * 80)
    print(f"Project: {SYSTEM_OVERVIEW['title']}")
    print(f"Mission: {SYSTEM_OVERVIEW['mission']}")
    print("=" * 80)
    print("\nCore Layers:")
    for layer, details in TECH_STACK_MATRIX.items():
        print(f"  - {layer}: {details}")
    print("\nArchitecture gotchas loaded:", len(ENGINEERING_GOTCHAS))
    print("RAGAS metrics supported:", len(EVALUATION_SUITE_SPEC["the_6_metrics"]))
    print("Manifest modules tracked:", len(CODE_MANIFEST))
    print("=" * 80)


if __name__ == "__main__":
    print_project_summary()
