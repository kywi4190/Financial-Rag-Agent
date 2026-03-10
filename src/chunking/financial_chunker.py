"""Structure-aware chunking for financial documents.

Splits filing sections into chunks that respect paragraph boundaries,
section headers, and financial table boundaries to preserve context.
"""

from src.chunking.models import DocumentChunk
from src.ingestion.models import FilingSection


class FinancialChunker:
    """Chunker that respects the structure of SEC filings.

    Args:
        chunk_size: Target token count per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
    """

    def __init__(self, chunk_size: int = 768, chunk_overlap: int = 128) -> None:
        """Initialize the chunker with size parameters."""
        ...

    def chunk_section(self, section: FilingSection) -> list[DocumentChunk]:
        """Split a filing section into document chunks.

        Args:
            section: A parsed filing section.

        Returns:
            List of DocumentChunk objects preserving structural boundaries.
        """
        ...

    def chunk_filing(self, sections: list[FilingSection]) -> list[DocumentChunk]:
        """Chunk all sections of a filing.

        Args:
            sections: All parsed sections from a single filing.

        Returns:
            Complete list of DocumentChunk objects for the filing.
        """
        ...
