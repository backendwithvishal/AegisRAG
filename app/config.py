import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL: Optional[str] = Field(
        default_factory=lambda: os.getenv("QDRANT_CLUSTER_ENDPOINT") or os.getenv("QDRANT_URL", "http://localhost:6333")
    )
    QDRANT_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    QDRANT_COLLECTION: str = Field(default="enterprise_rag")

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_FALLBACK_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_FALLBACK_API_KEY"))
    JUDGE_GROQ: Optional[str] = Field(default_factory=lambda: os.getenv("JUDGE_GROQ"))

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("PORTKEY_API_KEY"))
    GROQ_SLUG: str = Field(default="rag")      # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2: str = Field(default="brag")   # fallback: @brag/llama-3.1-8b-instant

    # --- OBSERVABILITY ---
    LOGFIRE_TOKEN: Optional[str] = Field(default_factory=lambda: os.getenv("LOGFIRE_TOKEN"))
    LANGSMITH_TRACING: str = Field(default_factory=lambda: os.getenv("LANGSMITH_TRACING", "true"))
    LANGSMITH_API_KEY: Optional[str] = Field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    LANGSMITH_PROJECT: str = Field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "rag_scale_test"))
    LANGSMITH_ENDPOINT: str = Field(default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))

    # --- PLATFORM CONFIGURATION [ADDED] ---
    CHECKPOINTER_TYPE: Literal["memory", "sqlite"] = Field(
        default_factory=lambda: os.getenv("CHECKPOINTER_TYPE", "memory")  # type: ignore
    )
    SQLITE_DB_PATH: str = Field(default_factory=lambda: os.getenv("SQLITE_DB_PATH", "state_checkpoint.db"))
    FEEDBACK_DB_PATH: str = Field(default_factory=lambda: os.getenv("FEEDBACK_DB_PATH", "feedback.db"))
    ENABLE_HYBRID_SEARCH: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() in ("true", "1", "yes")
    )
    MAX_CONTEXT_CHARS: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "8000"))
    )
    API_KEY_AUTH: Optional[str] = Field(default_factory=lambda: os.getenv("AEGIS_API_KEY"))
    RATE_LIMIT_PER_MINUTE: int = Field(default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))

    def get_collection_name(self, dimension: int = 3072) -> str:
        """
        [FIX] Returns a collection name with dimension suffix to prevent dimension clashes
        when switching between Gemini (3072-dim) and SentenceTransformers (768-dim).
        """
        base = self.QDRANT_COLLECTION
        if f"_{dimension}" in base:
            return base
        return f"{base}_{dimension}"


# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
