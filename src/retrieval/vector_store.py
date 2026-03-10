"""ChromaDB vector store wrapper.

Provides a clean interface for indexing document chunks and
performing dense vector similarity search using OpenAI embeddings.
"""

import logging
from typing import Optional

import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.chunking.models import ChunkMetadata, DocumentChunk
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

BATCH_SIZE = 2048


class ChromaStore:
    """Wrapper around ChromaDB for dense vector search using OpenAI embeddings.

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
        """Initialize ChromaDB client and OpenAI embedding model."""
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed_model = OpenAIEmbedding(model_name=embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches of up to 2048."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = self._call_embed_api(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIConnectionError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_embed_api(self, texts: list[str]) -> list[list[float]]:
        """Call OpenAI embedding API for a single batch with retry."""
        logger.debug("Embedding batch of %d texts", len(texts))
        return self._embed_model.get_text_embedding_batch(texts)

    @staticmethod
    def _flatten_metadata(meta: ChunkMetadata) -> dict:
        """Flatten ChunkMetadata to a dict for ChromaDB storage."""
        return {
            "ticker": meta.ticker,
            "year": meta.year,
            "filing_type": meta.filing_type,
            "section_name": meta.section_name,
            "chunk_index": meta.chunk_index,
            "is_table": meta.is_table,
            "page_estimate": meta.page_estimate,
        }

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Batch embed and store chunks with metadata.

        Args:
            chunks: List of DocumentChunk objects to index.
        """
        if not chunks:
            return

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = self._embed(texts)

            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[self._flatten_metadata(c.metadata) for c in batch],
            )
            logger.info(
                "Indexed batch %d-%d of %d chunks", i, i + len(batch), len(chunks)
            )

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Perform dense vector similarity search.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            filters: Optional dict of metadata filters (e.g. {"ticker": "AAPL"}).

        Returns:
            List of SearchResult objects ranked by cosine similarity.
        """
        count = self._collection.count()
        if count == 0:
            return []

        query_embedding = self._embed([query])[0]
        where = self._build_where(filters) if filters else None
        effective_k = min(top_k, count)

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": effective_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                distance = results["distances"][0][idx]
                meta_dict = results["metadatas"][0][idx]
                search_results.append(
                    SearchResult(
                        chunk_id=results["ids"][0][idx],
                        content=results["documents"][0][idx],
                        score=1.0 - distance,
                        metadata=ChunkMetadata(**meta_dict),
                        source="dense",
                    )
                )

        return search_results

    @staticmethod
    def _build_where(filters: dict) -> dict | None:
        """Build a ChromaDB where clause from a filter dict."""
        conditions = [{k: {"$eq": v}} for k, v in filters.items() if v is not None]
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def delete_by_ticker(self, ticker: str) -> None:
        """Remove all chunks for a company.

        Args:
            ticker: Stock ticker symbol to delete.
        """
        self._collection.delete(where={"ticker": {"$eq": ticker}})
        logger.info("Deleted all chunks for ticker %s", ticker)

    def get_collection_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dict with count, tickers, and years available.
        """
        count = self._collection.count()
        if count == 0:
            return {"count": 0, "tickers": [], "years": []}

        all_data = self._collection.get(include=["metadatas"])
        tickers: set[str] = set()
        years: set[int] = set()
        for meta in all_data["metadatas"]:
            if "ticker" in meta:
                tickers.add(meta["ticker"])
            if "year" in meta:
                years.add(meta["year"])

        return {
            "count": count,
            "tickers": sorted(tickers),
            "years": sorted(years),
        }

    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB."""
        self._client.delete_collection(self._collection_name)
        logger.info("Deleted collection %s", self._collection_name)
