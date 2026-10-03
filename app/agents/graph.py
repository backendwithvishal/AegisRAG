import os
import sqlite3
import logfire
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)


# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState) -> str:
    """
    Routes the workflow based on the planner's decision:
      - 'CONVERSATIONAL' → directly to responder (memory recall)
      - Search query string → to retriever (vector DB + rerank)
    """
    if state.get("current_query") == "CONVERSATIONAL":
        return "responder"
    return "retriever"


workflow.set_entry_point("planner")

# Conditional Edge: Planner -> route_planner -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# --- CHECKPOINTER CONFIGURATION [ADDED] ---
def _build_checkpointer():
    if settings.CHECKPOINTER_TYPE == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            conn = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
            logfire.info(f"💾 Persistent SQLite checkpointer active: {settings.SQLITE_DB_PATH}")
            return SqliteSaver(conn)
        except Exception as e:
            logfire.warning(f"⚠️ SQLite checkpointer initialization failed: {e}. Falling back to MemorySaver.")
            return MemorySaver()
    return MemorySaver()


checkpointer = _build_checkpointer()

# 4. Compile the Graph with Checkpointer
rag_agent = workflow.compile(checkpointer=checkpointer)
