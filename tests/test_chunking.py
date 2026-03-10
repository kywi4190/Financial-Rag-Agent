"""Tests for the chunking package (financial chunker and table handler)."""

from datetime import date

import pytest
import tiktoken

from src.chunking.financial_chunker import FinancialChunker
from src.chunking.models import ChunkMetadata, DocumentChunk
from src.chunking.table_handler import detect_tables, html_table_to_markdown, is_financial_table
from src.ingestion.models import FilingMetadata, FilingSection

_ENCODING = tiktoken.get_encoding("cl100k_base")

_DEFAULT_METADATA = FilingMetadata(
    cik="0000320193",
    company_name="Apple Inc.",
    ticker="AAPL",
    filing_type="10-K",
    filing_date=date(2024, 10, 31),
    accession_number="0000320193-24-000001",
)


def _make_section(
    content: str,
    section_name: str = "Item 1A. Risk Factors",
    tables: list[str] | None = None,
) -> FilingSection:
    return FilingSection(
        section_name=section_name,
        content=content,
        metadata=_DEFAULT_METADATA,
        tables=tables or [],
    )


# ---------------------------------------------------------------------------
# FinancialChunker tests
# ---------------------------------------------------------------------------


class TestFinancialChunker:
    """Tests for FinancialChunker."""

    def test_section_boundary_splitting(self) -> None:
        """Chunks from different sections never share content."""
        s1 = _make_section(
            "Alpha section one paragraph.\n\nBeta section one paragraph.",
            section_name="Item 1. Business",
        )
        s2 = _make_section(
            "Gamma section two paragraph.\n\nDelta section two paragraph.",
            section_name="Item 1A. Risk Factors",
        )

        chunker = FinancialChunker(chunk_size=768, chunk_overlap=128)
        chunks = chunker.chunk_filing([s1, s2])

        section_names = {c.metadata.section_name for c in chunks}
        assert "Item 1. Business" in section_names
        assert "Item 1A. Risk Factors" in section_names

        s2_chunks = [c for c in chunks if c.metadata.section_name == "Item 1A. Risk Factors"]
        for c in s2_chunks:
            assert "Alpha section one" not in c.content
            assert "Beta section one" not in c.content

    def test_table_never_split(self) -> None:
        """Tables are always emitted as a single atomic chunk."""
        large_table = "<table>" + "".join(
            f"<tr><td>Row {i}</td><td>${i * 1000:,}</td><td>Revenue line {i}</td></tr>"
            for i in range(50)
        ) + "</table>"

        section = _make_section("Text before table.", tables=[large_table])
        chunker = FinancialChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_section(section)

        table_chunks = [c for c in chunks if c.metadata.is_table]
        assert len(table_chunks) == 1
        # Verify the table content is fully present
        assert "Row 0" in table_chunks[0].content
        assert "Row 49" in table_chunks[0].content

    def test_metadata_prefix_format(self) -> None:
        """Metadata prefix follows '[Ticker: X] [Year: Y] ...' format."""
        section = _make_section("Some content here.")
        chunker = FinancialChunker()
        chunks = chunker.chunk_section(section)

        assert len(chunks) >= 1
        chunk = chunks[0]
        expected = "[Ticker: AAPL] [Year: 2024] [Section: Item 1A. Risk Factors] [Filing: 10-K]"
        assert chunk.metadata_prefix == expected
        assert chunk.content.startswith(expected)

    def test_chunk_size_limits(self) -> None:
        """Text chunks should not exceed the configured chunk_size."""
        paragraphs = [
            f"Paragraph {i} contains enough filler words to be meaningful in testing. " * 5
            for i in range(20)
        ]
        content = "\n\n".join(paragraphs)

        section = _make_section(content)
        chunker = FinancialChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_section(section)

        for chunk in chunks:
            if not chunk.metadata.is_table:
                # Allow a small tolerance for rounding at sentence boundaries
                assert chunk.token_count <= chunker.chunk_size + 15, (
                    f"Chunk {chunk.metadata.chunk_index} has {chunk.token_count} tokens "
                    f"(limit {chunker.chunk_size})"
                )

    def test_overlap_at_paragraph_boundaries(self) -> None:
        """Overlap content consists of complete paragraphs, never partial text."""
        paragraphs = [f"Unique paragraph {chr(65 + i)} with distinctive words." for i in range(15)]
        content = "\n\n".join(paragraphs)

        section = _make_section(content)
        chunker = FinancialChunker(chunk_size=60, chunk_overlap=20)
        chunks = chunker.chunk_section(section)
        text_chunks = [c for c in chunks if not c.metadata.is_table]

        assert len(text_chunks) > 1, "Need multiple chunks to test overlap"

        # Every text fragment in a chunk body must be a complete original paragraph
        for chunk in text_chunks:
            body = chunk.content.split("\n\n", 1)[-1]  # strip prefix
            chunk_paras = [p.strip() for p in body.split("\n\n") if p.strip()]
            for cp in chunk_paras:
                assert cp in paragraphs, f"Partial paragraph found: {cp[:60]}..."

        # Consecutive chunks share at least one paragraph (overlap exists)
        for i in range(len(text_chunks) - 1):
            body_a = text_chunks[i].content.split("\n\n", 1)[-1]
            body_b = text_chunks[i + 1].content.split("\n\n", 1)[-1]
            paras_a = {p.strip() for p in body_a.split("\n\n") if p.strip()}
            paras_b = {p.strip() for p in body_b.split("\n\n") if p.strip()}
            assert paras_a & paras_b, (
                f"No overlap between chunk {i} and {i + 1}"
            )

    def test_token_counting_accuracy(self) -> None:
        """Token counts match independent tiktoken calculation."""
        section = _make_section(
            "This is a test of token counting accuracy. It should be precise."
        )
        chunker = FinancialChunker()
        chunks = chunker.chunk_section(section)

        for chunk in chunks:
            expected = len(_ENCODING.encode(chunk.content))
            assert chunk.token_count == expected

    def test_empty_content_produces_no_text_chunks(self) -> None:
        """A section with empty content but tables still produces table chunks."""
        table = "<table><tr><td>Revenue</td><td>$1,000</td></tr></table>"
        section = _make_section("", tables=[table])
        chunker = FinancialChunker()
        chunks = chunker.chunk_section(section)

        text_chunks = [c for c in chunks if not c.metadata.is_table]
        table_chunks = [c for c in chunks if c.metadata.is_table]
        assert len(text_chunks) == 0
        assert len(table_chunks) == 1

    def test_chunk_indices_are_sequential(self) -> None:
        """chunk_index values are sequential within a section."""
        paragraphs = [f"Paragraph {i} with some words." for i in range(10)]
        section = _make_section("\n\n".join(paragraphs))
        chunker = FinancialChunker(chunk_size=60, chunk_overlap=10)
        chunks = chunker.chunk_section(section)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i


# ---------------------------------------------------------------------------
# TableHandler tests
# ---------------------------------------------------------------------------


class TestTableHandler:
    """Tests for table handler functions."""

    def test_detect_tables_finds_html_tables(self) -> None:
        """detect_tables extracts all <table> elements."""
        html = (
            "<p>Text</p>"
            "<table><tr><td>A</td></tr></table>"
            "<p>More</p>"
            "<table><tr><td>B</td></tr></table>"
        )
        tables = detect_tables(html)
        assert len(tables) == 2
        assert "A" in tables[0]
        assert "B" in tables[1]

    def test_detect_tables_empty_input(self) -> None:
        """detect_tables returns empty list for content without tables."""
        assert detect_tables("<p>No tables here</p>") == []

    def test_html_table_to_markdown_formats_correctly(self) -> None:
        """HTML tables are converted to pipe-delimited markdown."""
        html = (
            "<table>"
            "<tr><th>Item</th><th>Value</th></tr>"
            "<tr><td>Revenue</td><td>$1,000</td></tr>"
            "<tr><td>Expenses</td><td>$800</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)

        assert "| Item | Value |" in md
        assert "| --- | --- |" in md
        assert "| Revenue | $1,000 |" in md
        assert "| Expenses | $800 |" in md

    def test_html_table_to_markdown_pads_uneven_rows(self) -> None:
        """Rows with fewer cells are padded to match the widest row."""
        html = (
            "<table>"
            "<tr><td>A</td><td>B</td><td>C</td></tr>"
            "<tr><td>X</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)
        lines = md.strip().split("\n")
        # All rows should have 3 columns
        for line in lines:
            assert line.count("|") == 4  # 3 columns = 4 pipes

    def test_is_financial_table_positive(self) -> None:
        """Tables with dollar amounts and financial terms are classified as financial."""
        table_md = (
            "| Item | Amount |\n"
            "| --- | --- |\n"
            "| Revenue | $1,234,567 |\n"
            "| Net Income | ($45,678) |"
        )
        assert is_financial_table(table_md) is True

    def test_is_financial_table_negative(self) -> None:
        """Decorative/navigation tables are not classified as financial."""
        table_md = (
            "| Link | Description |\n"
            "| --- | --- |\n"
            "| Home | Main page |\n"
            "| About | Info page |"
        )
        assert is_financial_table(table_md) is False
