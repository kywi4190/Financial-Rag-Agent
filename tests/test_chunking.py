"""Tests for the chunking package (financial chunker and table handler)."""

import pytest


class TestFinancialChunker:
    """Tests for FinancialChunker."""

    def test_chunk_section_respects_size_limit(self) -> None:
        """Test that chunks do not exceed the configured chunk_size."""
        ...

    def test_chunk_section_preserves_overlap(self) -> None:
        """Test that consecutive chunks have the expected overlap."""
        ...

    def test_chunk_filing_processes_all_sections(self) -> None:
        """Test that chunk_filing produces chunks for every section."""
        ...


class TestTableHandler:
    """Tests for TableHandler."""

    def test_extract_tables_finds_html_tables(self) -> None:
        """Test that extract_tables correctly identifies table elements."""
        ...

    def test_table_to_markdown_formats_correctly(self) -> None:
        """Test that table_to_markdown produces valid markdown tables."""
        ...
