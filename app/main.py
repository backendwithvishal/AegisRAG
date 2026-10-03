# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import internal and third-party modules
import time
import json
import sqlite3
from typing import Optional, List, Dict, Any, Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response, Request, HTTPException, Security, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from app.config import settings
from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard, redact_pii
from app.services.retrieval.embedding import get_model_type, get_embedding_dim


# Database helper for user feedback
def init_feedback_db():
    try:
        conn = sqlite3.connect(settings.FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logfire.error(f"Failed to init feedback db: {e}")


# Initialize feedback DB immediately on module load
init_feedback_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logfire.info("🚀 Starting AegisRAG API service...")
    initialize_rails()
    init_feedback_db()
    yield
    logfire.info("🛑 Shutting down AegisRAG API service...")


# Initialize FastAPI with OpenAPI metadata and lifespan
app = FastAPI(
    title="AegisRAG - Enterprise Agentic RAG Platform",
    description="Fault-tolerant, high-throughput QA system over enterprise docs (Kubernetes, Intel hardware, networking).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for enterprise API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if settings.API_KEY_AUTH:
        if not api_key or api_key != settings.API_KEY_AUTH:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header"
            )
    return api_key


# ── Pydantic Request & Response Models ─────────────────────────────────────────

class QueryRequest(BaseModel):
    q: str = Field(..., description="The query or prompt string", json_schema_extra={"example": "How do you start Redis for a Kubernetes work queue?"})
    thread_id: Optional[str] = Field(default="default_user", description="Session/thread identifier for conversational memory", json_schema_extra={"example": "user_session_101"})


class QueryResponse(BaseModel):
    question: str
    answer: str
    thought_process: List[str]
    status: str
    sources: List[str]


class FeedbackRequest(BaseModel):
    thread_id: str = Field(..., json_schema_extra={"example": "user_session_101"})
    rating: Literal["up", "down"] = Field(..., json_schema_extra={"example": "up"})
    comment: Optional[str] = Field(None, json_schema_extra={"example": "Accurate command reference."})


# ── Core Endpoints ─────────────────────────────────────────────────────────────

@app.get("/", summary="Root Health Check")
def home():
    return {
        "service": "AegisRAG Enterprise Agentic RAG API",
        "status": "live",
        "embedding_backend": get_model_type(),
        "embedding_dim": get_embedding_dim()
    }


@app.get("/health", summary="Service Liveness Probe")
def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "AegisRAG"
    }


@app.get("/ready", summary="Service Readiness Probe")
def ready():
    from app.services.retrieval.qdrant_service import get_qdrant_client
    checks = {"qdrant": False, "guardrails": False}
    try:
        client = get_qdrant_client()
        client.get_collections()
        checks["qdrant"] = True
    except Exception as e:
        logfire.warning(f"Readiness check failed for Qdrant: {e}")

    try:
        from app.guardrails.rails import _rails
        checks["guardrails"] = _rails is not None or settings.GROQ_API_KEY is not None
    except Exception:
        checks["guardrails"] = False

    is_ready = checks["qdrant"] and checks["guardrails"]
    return {
        "status": "ready" if is_ready else "degraded",
        "checks": checks,
        "timestamp": time.time()
    }


@app.get("/graph", summary="Agent Workflow Diagram")
def get_graph_image():
    """
    Returns the Mermaid image / representation of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        logfire.warning(f"Could not render Mermaid PNG: {e}. Returning Mermaid syntax.")
        try:
            mermaid_str = rag_agent.get_graph().draw_mermaid()
            return {"mermaid": mermaid_str}
        except Exception as inner_e:
            return {"error": f"Could not generate graph: {inner_e}"}


@app.post("/query", response_model=QueryResponse, summary="Execute Agentic RAG Query", dependencies=[Depends(verify_api_key)])
def query(request: QueryRequest):
    """
    Executes the 2-Gate Agentic RAG lifecycle:
      - Gate 1: NeMo Guardrails fast intent filter (blocks off-topic, jailbreaks, answers dialog)
      - Gate 2: LangGraph StateGraph (Planner classification -> 2-stage Retrieval -> Grounded Responder)
    """
    q = request.q
    thread_id = request.thread_id or "default_user"
    redacted_preview = redact_pii(q[:80])

    with logfire.span("📬 POST /query Lifecycle", query=redacted_preview, thread_id=thread_id):
        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph..."
        }

        # Configuration for Memory Saver (thread_id)
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Gate 1: NeMo Guardrails
            rail_fired, rail_response = guard(q)
            if rail_fired:
                logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
                return QueryResponse(
                    question=q,
                    answer=rail_response or "",
                    thought_process=["Intent: Guardrails Fired", "Retrieval: Skipped"],
                    status="Blocked by guardrails.",
                    sources=[]
                )

            # Gate 2: LangGraph RAG pipeline
            final_output = rag_agent.invoke(initial_state, config=config)

            return QueryResponse(
                question=q,
                answer=final_output.get("final_answer") or "",
                thought_process=final_output.get("plan") or [],
                status=final_output.get("status") or "Completed",
                sources=final_output.get("documents") or []
            )
        except Exception as e:
            logfire.error(f"❌ Backend Execution Failed: {e}")
            return QueryResponse(
                question=q,
                answer="I apologize, but I encountered an internal error while processing your request. Please try again later.",
                thought_process=["Error encountered during execution."],
                status="error",
                sources=[]
            )


@app.post("/query/stream", summary="Stream Agentic RAG Response (SSE)", dependencies=[Depends(verify_api_key)])
def query_stream(request: QueryRequest):
    """
    [ADDED] Server-Sent Events (SSE) streaming endpoint for responsive token delivery.
    """
    q = request.q
    thread_id = request.thread_id or "default_user"

    def event_generator():
        # Gate 1: NeMo Guardrails
        rail_fired, rail_response = guard(q)
        if rail_fired:
            payload = {
                "type": "block",
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "token": rail_response,
                "sources": []
            }
            yield f"data: {json.dumps(payload)}\n\n"
            return

        # Gate 2: Execute LangGraph
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "current_query": q,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph..."
        }
        
        final_output = rag_agent.invoke(initial_state, config=config)
        answer = final_output.get("final_answer") or ""
        
        # Stream chunks of tokens
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            payload = {
                "type": "token",
                "token": chunk,
                "thought_process": final_output.get("plan", []),
                "status": final_output.get("status", ""),
                "sources": final_output.get("documents", []) if i == len(words) - 1 else []
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.015)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/feedback", summary="Record User Feedback", dependencies=[Depends(verify_api_key)])
def record_feedback(feedback: FeedbackRequest):
    """
    [ADDED] Stores thumbs-up / thumbs-down evaluation feedback with comments.
    """
    try:
        init_feedback_db()
        conn = sqlite3.connect(settings.FEEDBACK_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (thread_id, rating, comment, timestamp) VALUES (?, ?, ?, ?)",
            (feedback.thread_id, feedback.rating, feedback.comment, time.time())
        )
        conn.commit()
        conn.close()
        logfire.info(f"👍 Feedback recorded | thread={feedback.thread_id}, rating={feedback.rating}")
        return {"status": "success", "message": "Feedback recorded successfully."}
    except Exception as e:
        logfire.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {e}")
