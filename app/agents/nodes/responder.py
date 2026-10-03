import logfire
from app.agents.state import AgentState
from app.config import settings
from app.gateway import portkey_client, extract_cache_status
from app.guardrails.rails import guard_output


def generate_node(state: AgentState) -> dict:
    """
    Synthesizes the final grounded response using Portkey Gateway (Llama 3.3 70B with 8B fallback).
    Surface Cache: Hit ⚡ if served from Portkey's gateway cache.
    Applies output safety guardrails to prevent credential/PII leakage.
    """
    query = state.get("current_query", "")

    history_str = ""
    for msg in state.get("messages", [])[:-1]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_str += f"{role}: {msg.get('content', '')}\n"

    user_msg = state["messages"][-1]["content"] if state.get("messages") else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
You are a knowledgeable and helpful Enterprise AI Assistant.
Answer the user's message accurately using ONLY the prior CONVERSATION HISTORY below.

CONVERSATION HISTORY:
{history_str if history_str.strip() else "(No prior history)"}

LATEST USER MESSAGE:
"{user_msg}"

Provide a concise, professional answer.
"""
    else:
        logfire.info("Generating technical RAG response grounded in documentation.")
        max_context_chars = settings.MAX_CONTEXT_CHARS
        full_context = ""

        for doc in state.get("documents", []):
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to maintain safe Groq TPM margins.")
                break

        prompt = f"""
You are a Senior Enterprise Technical Architect specializing in Kubernetes, Intel Hardware, and Networking.
Answer the user's question accurately using ONLY the TECHNICAL CONTEXT provided below.

RULES:
1. Ground your answer strictly in the TECHNICAL CONTEXT. Do not invent details or speculate.
2. If the context does not contain sufficient facts to answer the question, clearly state: "I apologize, but this information is not found in the enterprise documentation."
3. Cite specific commands, configurations, or parameters when present in the context.

TECHNICAL CONTEXT:
{full_context if full_context.strip() else "(No relevant technical documentation found.)"}

CONVERSATION HISTORY:
{history_str if history_str.strip() else "(No prior history)"}

USER QUESTION:
"{user_msg}"
"""

    with logfire.span("✍️ LLM Synthesis (Portkey Gateway)"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            raw_content = response.choices[0].message.content or ""
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            # Output Guardrail: scan for secrets/PII
            _, sanitized_content = guard_output(raw_content)

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state.get("plan", []) + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM Gateway.")
                plan_update = state.get("plan", [])
                status = "Response generated."

            return {
                "final_answer": sanitized_content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": sanitized_content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
