"""Pydantic models for search and retrieval results.

Defines the structured representations of search results from
vector search, BM25, hybrid fusion, and reranking stages.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.chunking.models import ChunkMetadata


class SearchResult(BaseModel):
    """A single search result from any retrieval stage.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        content: Text content of the matched chunk.
        score: Relevance score (interpretation depends on source).
        metadata: Chunk metadata for attribution and filtering.
        source: Which retrieval stage produced this result.
    """

    chunk_id: str
    content: str
    score: float
    metadata: ChunkMetadata
    source: str = "dense"


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline.

    Attributes:
        top_k: Number of candidates from each retrieval source.
        rerank_top_k: Number of results after reranking.
        dense_weight: Weight for dense vector search in hybrid fusion.
        sparse_weight: Weight for sparse BM25 search in hybrid fusion.
        filters: Optional metadata filters applied to search.
    """

    top_k: int = 20
    rerank_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    filters: Optional[dict] = None


class RetrievalResult(BaseModel):
    """Aggregated result from the full retrieval pipeline.

    Attributes:
        query: The original query string.
        results: Ordered list of search results after all stages.
        stages_applied: List of retrieval stages applied (e.g., ['vector', 'bm25', 'rrf', 'rerank']).
    """

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    stages_applied: list[str] = Field(default_factory=list)
