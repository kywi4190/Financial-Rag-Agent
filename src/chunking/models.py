"""Pydantic models for document chunks.

Defines the structured representation of text chunks produced by
the financial chunker and table handler.
"""

from uuid import uuid4

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to a single chunk.

    Attributes:
        ticker: Stock ticker symbol.
        year: Fiscal year of the filing.
        filing_type: SEC form type (e.g., '10-K').
        section_name: Name of the filing section this chunk came from.
        chunk_index: Position of this chunk within its section.
        is_table: Whether this chunk contains a financial table.
        page_estimate: Approximate page number in the original filing.
    """

    ticker: str
    year: int
    filing_type: str
    section_name: str
    chunk_index: int
    is_table: bool = False
    page_estimate: int = 1


class DocumentChunk(BaseModel):
    """A single chunk of text ready for embedding and indexing.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        content: The chunk text content (includes metadata prefix).
        metadata: Structured metadata for filtering and attribution.
        metadata_prefix: The metadata prefix string prepended to content.
        token_count: Token count of the full content (via tiktoken cl100k_base).
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    metadata: ChunkMetadata
    metadata_prefix: str = ""
    token_count: int = 0
