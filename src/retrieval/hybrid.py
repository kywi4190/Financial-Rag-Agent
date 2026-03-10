"""Reciprocal Rank Fusion for hybrid search.

Combines results from dense vector search and BM25 sparse search
using weighted Reciprocal Rank Fusion (RRF) to produce a unified ranking.
Numerical queries automatically boost table/XBRL chunk scores.
"""

import logging
import re
from collections import defaultdict
from typing import Optional

from src.retrieval.bm25_search import BM25Index
from src.retrieval.models import RetrievalConfig, SearchResult
from src.retrieval.vector_store import ChromaStore

logger = logging.getLogger(__name__)

_NUMERICAL_PATTERNS = [
    r"\d+",
    r"how much",
    r"how many",
    r"what was the",
    r"what is the",
    r"revenue",
    r"earnings",
    r"income",
    r"profit",
    r"ebitda",
    r"eps",
    r"margin",
    r"ratio",
    r"debt",
    r"assets",
    r"liabilities",
    r"cash flow",
    r"growth",
    r"compare",
    r"year over year",
    r"yoy",
    r"\$",
    r"%",
]

_NUMERICAL_RE = re.compile("|".join(_NUMERICAL_PATTERNS), re.IGNORECASE)
_TABLE_BOOST = 1.5


class HybridRetriever:
    """Combines vector and BM25 search results via weighted Reciprocal Rank Fusion.

    Args:
        vector_store: ChromaStore instance for dense retrieval.
        bm25_index: BM25Index instance for sparse retrieval.
        config: RetrievalConfig with top_k, weights, and filters.
    """

    def __init__(
        self,
        vector_store: ChromaStore,
        bm25_index: BM25Index,
        config: RetrievalConfig | None = None,
    ) -> None:
        """Initialize the hybrid retriever."""
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._config = config or RetrievalConfig()
        self._rrf_k = 60

    @staticmethod
    def _is_numerical_query(query: str) -> bool:
        """Detect whether a query is numerical/quantitative.

        Returns True when two or more numerical signal patterns match.
        """
        return len(_NUMERICAL_RE.findall(query)) >= 2

    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Run hybrid retrieval with RRF fusion and numerical boosting.

        Args:
            query: Natural language query string.
            filters: Optional metadata filters forwarded to the vector store.

        Returns:
            Fused list of SearchResult objects ranked by RRF score.
        """
        cfg = self._config
        effective_filters = filters or cfg.filters

        dense_results = self._vector_store.search(
            query, top_k=cfg.top_k, filters=effective_filters
        )
        sparse_results = self._bm25_index.search(query, top_k=cfg.top_k)

        fused = self._rrf_fuse(
            dense_results, sparse_results, cfg.dense_weight, cfg.sparse_weight
        )

        if self._is_numerical_query(query):
            fused = [
                r.model_copy(update={"score": r.score * _TABLE_BOOST})
                if r.metadata.is_table
                else r
                for r in fused
            ]
            fused.sort(key=lambda r: r.score, reverse=True)

        return fused[: cfg.top_k]

    def _rrf_fuse(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        dense_weight: float,
        sparse_weight: float,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion: score = sum(weight / (k + rank))."""
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            scores[result.chunk_id] += dense_weight / (self._rrf_k + rank)
            if (
                result.chunk_id not in docs
                or result.score > docs[result.chunk_id].score
            ):
                docs[result.chunk_id] = result

        for rank, result in enumerate(sparse_results, start=1):
            scores[result.chunk_id] += sparse_weight / (self._rrf_k + rank)
            if (
                result.chunk_id not in docs
                or result.score > docs[result.chunk_id].score
            ):
                docs[result.chunk_id] = result

        return [
            docs[cid].model_copy(update={"score": scores[cid], "source": "hybrid"})
            for cid in sorted(scores, key=lambda cid: scores[cid], reverse=True)
        ]
