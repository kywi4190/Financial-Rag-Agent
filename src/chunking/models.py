"""Pydantic models for document chunks.

Defines the structured representation of text chunks produced by
the financial chunker and table handler.
"""

from typing import Optional

from pydantic import BaseModel, Field

from src.ingestion.models import FilingMetadata


class ChunkMetadata(BaseModel):
    """Metadata attached to a single chunk.

    Attributes:
        filing: Source filing metadata.
        section_name: Name of the filing section this chunk came from.
        chunk_index: Position of this chunk within its section.
        is_table: Whether this chunk contains a financial table.
        page_number: Approximate page number in the original filing.
    """

    filing: FilingMetadata
    section_name: str
    chunk_index: int
    is_table: bool = False
    page_number: Optional[int] = None


class DocumentChunk(BaseModel):
    """A single chunk of text ready for embedding and indexing.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: The chunk text content.
        metadata: Structured metadata for filtering and attribution.
        token_count: Approximate token count of the text.
    """

    chunk_id: str
    text: str
    metadata: ChunkMetadata
    token_count: int = 0
