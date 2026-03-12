"""BM25 sparse keyword search.

Provides term-frequency-based retrieval using the Okapi BM25 algorithm
as a complement to dense vector search in the hybrid pipeline.
"""

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking.models import DocumentChunk
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 sparse search index over document chunks.

    Maintains an in-memory BM25 index that can be rebuilt
    whenever the document corpus changes.
    """

    def __init__(self) -> None:
        """Initialize the BM25 search index."""
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize preserving financial compound terms.

        Keeps SEC filing types (10-K, 10-Q), dollar amounts, percentages,
        and hyphenated compound words as single tokens.
        """
        text = text.lower()
        tokens = re.findall(
            r"\d{1,2}-[kq]"            # SEC filings: 10-k, 10-q
            r"|\$?\d+(?:\.\d+)?%?"     # numbers with optional $ prefix or % suffix
            r"|[a-z]+(?:-[a-z]+)+"     # hyphenated compounds: year-over-year
            r"|[a-z0-9]+",             # regular alphanumeric tokens
            text,
        )
        return [cleaned for t in tokens if (cleaned := t.lstrip("$").rstrip("%"))]

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from a set of chunks.

        Args:
            chunks: List of DocumentChunk objects to index.
        """
        self._chunks = list(chunks)
        tokenized = [self._tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("Built BM25 index over %d chunks", len(self._chunks))

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Perform BM25 keyword search.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects ranked by BM25 score.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=float(scores[idx]),
                    metadata=chunk.metadata,
                    source="sparse",
                )
            )
        return results
