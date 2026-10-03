import logfire
from app.agents.state import AgentState
from app.gateway import get_langchain_llm

llm = get_langchain_llm(feature="planner")


def planner_node(state: AgentState) -> dict:
    """
    The Planner determines if a knowledge search is needed or if the query can be
    resolved conversationally via thread memory.
    """
    # Build conversation history from prior turns
    history = ""
    for msg in state.get("messages", [])[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history += f"{role}: {msg.get('content', '')}\n"

    user_message = state["messages"][-1]["content"] if state.get("messages") else ""

    prompt = f"""
You are an intelligent Enterprise Assistant Planner.
Analyze the conversation history and the latest user message.

CONVERSATION HISTORY:
{history if history.strip() else "(No prior conversation)"}

LATEST MESSAGE:
"{user_message}"

Task:
1. If the latest message is a conversational greeting, farewell, or asks about prior conversation context (e.g., "what was my first question?", "repeat what you just said"), respond with 'CONVERSATIONAL'.
2. If it is a technical question about Kubernetes, Intel hardware, or Networking that requires fresh documentation, output a concise standalone search query for vector retrieval.

Output ONLY 'CONVERSATIONAL' or the synthesized search query without any explanation.
"""

    with logfire.span("🧠 Planner Decision"):
        try:
            decision = llm.invoke(prompt).content.strip()
            # Clean possible markdown quotes or backticks
            decision = decision.strip('"`\'')
            logfire.info(f"Planner intent identified: {decision}")
        except Exception as e:
            logfire.error(f"Planner LLM failed: {e}. Defaulting to user message query.")
            decision = user_message

    if decision.upper() == "CONVERSATIONAL" or "CONVERSATIONAL" in decision.upper():
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
