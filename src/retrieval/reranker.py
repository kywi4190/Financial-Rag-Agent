"""Cross-encoder reranking for search results.

Uses a cross-encoder model from sentence-transformers to
re-score and reorder search results for improved precision.
Scores are normalized to [0, 1] via min-max scaling.
"""

import logging

from sentence_transformers import CrossEncoder

from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks search results using a cross-encoder model.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> None:
        """Initialize the cross-encoder reranker."""
        self._model = CrossEncoder(model_name)
        logger.info("Loaded cross-encoder model: %s", model_name)

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
            Reranked and truncated list of SearchResult objects
            with scores normalized to [0, 1].
        """
        if not results:
            return []

        pairs = [(query, r.content) for r in results]

        try:
            raw_scores = self._model.predict(pairs)
        except Exception:
            logger.exception(
                "Cross-encoder prediction failed, returning un-reranked results"
            )
            return results[:top_k]

        min_s = float(min(raw_scores))
        max_s = float(max(raw_scores))
        span = max_s - min_s

        reranked = [
            r.model_copy(
                update={
                    "score": (float(raw_scores[i]) - min_s) / span if span > 0 else 1.0,
                    "source": "rerank",
                }
            )
            for i, r in enumerate(results)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:top_k]
