import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Production gateway config:
#   - Fallback: primary @rag/llama-3.3-70b-versatile → @brag/llama-3.1-8b-instant on failure
#   - Cache: simple mode (exact match) / semantic mode on Enterprise tier
#   - Retry: 2 attempts on rate limit (429) or server error (503) before triggering fallback
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "request_timeout": 60000,  # 60s hard timeout in ms
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/llama-3.1-8b-instant"}},
    ]
}

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=GATEWAY_CONFIG
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is an AI Gateway exposing an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq connects directly to Groq without gateway mediation.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + fallback/cache config). The @rag/model-name slug format is Portkey-specific.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY or "dummy_portkey_key",
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY or "",
            config=GATEWAY_CONFIG,
            metadata={
                "feature": feature,
                "_user": "aegis-rag",
                "environment": "production"
            }
        )
    )


def extract_cache_status(response) -> str:
    """
    Defensively extract x-portkey-cache-status from Portkey response headers.
    Returns 'HIT' or 'MISS'.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            headers = getattr(raw, "headers", {})
            status = headers.get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"