"""Tests for the ingestion package (EDGAR client and filing parser)."""

import pytest


class TestEdgarClient:
    """Tests for EdgarClient."""

    def test_get_filings_returns_metadata(self) -> None:
        """Test that get_filings returns a list of FilingMetadata."""
        ...

    def test_download_filing_returns_html(self) -> None:
        """Test that download_filing returns raw HTML content."""
        ...


class TestFilingParser:
    """Tests for FilingParser."""

    def test_parse_sections_extracts_items(self) -> None:
        """Test that parse_sections identifies standard 10-K items."""
        ...

    def test_extract_xbrl_facts_returns_facts(self) -> None:
        """Test that extract_xbrl_facts parses inline XBRL tags."""
        ...
