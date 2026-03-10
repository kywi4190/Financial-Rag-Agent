"""ChromaDB vector store wrapper.

Provides a clean interface for indexing document chunks and
performing dense vector similarity search using OpenAI embeddings.
"""

from src.chunking.models import DocumentChunk
from src.retrieval.models import SearchResult


class VectorStore:
    """Wrapper around ChromaDB for dense vector search.

    Args:
        persist_dir: Directory for ChromaDB persistence.
        collection_name: Name of the ChromaDB collection.
        embedding_model: OpenAI embedding model identifier.
    """

    def __init__(
        self,
        persist_dir: str = ".chroma",
        collection_name: str = "financial_filings",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the vector store."""
        ...

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Index a batch of document chunks.

        Args:
            chunks: List of DocumentChunk objects to index.
        """
        ...

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Perform dense vector similarity search.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects ranked by similarity.
        """
        ...

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB."""
        ...
