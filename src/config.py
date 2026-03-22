"""Central configuration using pydantic-settings.

Reads environment variables from a .env file and provides typed,
validated configuration to all modules in the project.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        openai_api_key: OpenAI API key for embeddings and LLM calls.
        sec_edgar_identity: User-agent identity string required by SEC EDGAR.
        chroma_persist_dir: Directory for ChromaDB persistence.
        embedding_model: OpenAI embedding model identifier.
        llm_model: OpenAI LLM model identifier.
        chunk_size: Token count per chunk for document splitting.
        chunk_overlap: Token overlap between consecutive chunks.
        top_k: Number of results to retrieve from hybrid search.
        rerank_top_k: Number of results to keep after cross-encoder reranking.
    """

    openai_api_key: str
    sec_edgar_identity: str
    chroma_persist_dir: str = ".chroma"
    bm25_persist_dir: str = ".bm25"
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    chunk_size: int = 768
    chunk_overlap: int = 128
    top_k: int = 10
    rerank_top_k: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


def get_settings() -> Settings:
    """Create and return a Settings instance.

    Returns:
        Validated Settings object with values from .env.
    """
    return Settings()
