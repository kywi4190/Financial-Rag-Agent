"""Chunking package — structure-aware document splitting for financial filings."""

from src.chunking.financial_chunker import FinancialChunker
from src.chunking.models import ChunkMetadata, DocumentChunk
from src.chunking.table_handler import detect_tables, html_table_to_markdown, is_financial_table

__all__ = [
    "ChunkMetadata",
    "DocumentChunk",
    "FinancialChunker",
    "detect_tables",
    "html_table_to_markdown",
    "is_financial_table",
]
