"""BM25 sparse keyword search.

Provides term-frequency-based retrieval using the Okapi BM25 algorithm
as a complement to dense vector search in the hybrid pipeline.
"""

from src.chunking.models import DocumentChunk
from src.retrieval.models import SearchResult


class BM25Search:
    """BM25 sparse search index over document chunks.

    Maintains an in-memory BM25 index that can be rebuilt
    whenever the document corpus changes.
    """

    def __init__(self) -> None:
        """Initialize the BM25 search index."""
        ...

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from a set of chunks.

        Args:
            chunks: List of DocumentChunk objects to index.
        """
        ...

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Perform BM25 keyword search.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects ranked by BM25 score.
        """
        ...
