"""Reciprocal Rank Fusion for hybrid search.

Combines results from dense vector search and BM25 sparse search
using Reciprocal Rank Fusion (RRF) to produce a unified ranking.
"""

from src.retrieval.models import RetrievalResult, SearchResult


class HybridRetriever:
    """Combines vector and BM25 search results via Reciprocal Rank Fusion.

    Args:
        rrf_k: RRF constant (default 60). Higher values reduce the
            impact of high-ranking documents.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        """Initialize the hybrid retriever."""
        ...

    def fuse(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Fuse two ranked lists using Reciprocal Rank Fusion.

        Args:
            vector_results: Results from dense vector search.
            bm25_results: Results from BM25 sparse search.
            top_k: Number of fused results to return.

        Returns:
            Merged and re-ranked list of SearchResult objects.
        """
        ...

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Args:
            query: Natural language query string.
            top_k: Number of final results to return.

        Returns:
            RetrievalResult with fused search results.
        """
        ...
