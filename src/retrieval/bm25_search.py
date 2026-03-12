"""BM25 sparse keyword search.

Provides term-frequency-based retrieval using the Okapi BM25 algorithm
as a complement to dense vector search in the hybrid pipeline.
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunking.models import ChunkMetadata, DocumentChunk
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

    def save_index(self, path: Path) -> None:
        """Serialize the BM25 index and chunk data to disk.

        Args:
            path: File path to save the index (e.g., .bm25/bm25_index.pkl).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks_data": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "metadata": c.metadata.model_dump(),
                }
                for c in self._chunks
            ],
            "tokenized_corpus": [self._tokenize(c.content) for c in self._chunks],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved BM25 index (%d chunks) to %s", len(self._chunks), path)

    @classmethod
    def load_index(cls, path: Path) -> "BM25Index":
        """Load a previously saved BM25 index from disk.

        Args:
            path: Path to the saved index file.

        Returns:
            Restored BM25Index ready for search.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        instance = cls()
        instance._chunks = [
            DocumentChunk(
                chunk_id=cd["chunk_id"],
                content=cd["content"],
                metadata=ChunkMetadata(**cd["metadata"]),
            )
            for cd in data["chunks_data"]
        ]
        instance._bm25 = BM25Okapi(data["tokenized_corpus"])
        logger.info("Loaded BM25 index (%d chunks) from %s", len(instance._chunks), path)
        return instance

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
