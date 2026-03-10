"""Cross-encoder reranking for search results.

Uses a cross-encoder model (e.g., from sentence-transformers) to
re-score and reorder search results for improved precision.
"""

from src.retrieval.models import SearchResult


class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Initialize the cross-encoder reranker."""
        ...

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results using the cross-encoder.

        Args:
            query: The original query string.
            results: List of SearchResult objects to rerank.
            top_k: Number of top results to keep after reranking.

        Returns:
            Reranked and truncated list of SearchResult objects.
        """
        ...
