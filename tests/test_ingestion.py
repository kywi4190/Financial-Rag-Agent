"""Tests for the ingestion package (EDGAR client and filing parser).

Unit tests mock the SEC EDGAR API; one integration test hits the real API.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.edgar_client import EdgarClient
from src.ingestion.filing_parser import FilingParser
from src.ingestion.models import FilingMetadata, FilingSection, XBRLFact


# ---------------------------------------------------------------------------
# EdgarClient tests
# ---------------------------------------------------------------------------
class TestEdgarClient:
    """Tests for EdgarClient."""

    IDENTITY = "TestBot test@example.com"

    def test_init_stores_identity(self) -> None:
        """EdgarClient stores the identity string for SEC requests."""
        client = EdgarClient(identity=self.IDENTITY)
        assert client.identity == self.IDENTITY

    # -- get_filings --------------------------------------------------------

    @patch("src.ingestion.edgar_client.Company")
    def test_download_10k(
        self, mock_company_cls: MagicMock, mock_edgar_filings_response: list[dict]
    ) -> None:
        """get_filings returns FilingMetadata list for 10-K filings."""
        mock_company = MagicMock()
        mock_company.get_filings.return_value.filter.return_value = (
            mock_edgar_filings_response
        )
        mock_company_cls.return_value = mock_company

        client = EdgarClient(identity=self.IDENTITY)
        results = client.get_filings(ticker="AAPL", filing_type="10-K", num_filings=2)

        assert isinstance(results, list)
        assert len(results) == 2
        for meta in results:
            assert isinstance(meta, FilingMetadata)
            assert meta.ticker == "AAPL"
            assert meta.filing_type == "10-K"

    @patch("src.ingestion.edgar_client.Company")
    def test_download_10q(self, mock_company_cls: MagicMock) -> None:
        """get_filings returns FilingMetadata list for 10-Q filings."""
        mock_company = MagicMock()
        mock_filing = MagicMock()
        mock_filing.cik = "0000320193"
        mock_filing.company_name = "Apple Inc."
        mock_filing.filing_date = "2024-08-02"
        mock_filing.accession_number = "0000320193-24-000099"
        mock_company.get_filings.return_value.filter.return_value = [mock_filing]
        mock_company_cls.return_value = mock_company

        client = EdgarClient(identity=self.IDENTITY)
        results = client.get_filings(ticker="AAPL", filing_type="10-Q", num_filings=1)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(m, FilingMetadata) for m in results)
        assert results[0].filing_type == "10-Q"

    @patch("src.ingestion.edgar_client.Company")
    def test_invalid_ticker_raises(self, mock_company_cls: MagicMock) -> None:
        """get_filings raises ValueError for a non-existent ticker."""
        mock_company_cls.side_effect = ValueError("Ticker not found: ZZZZZZ")

        client = EdgarClient(identity=self.IDENTITY)
        with pytest.raises(ValueError, match="Ticker not found"):
            client.get_filings(ticker="ZZZZZZ")

    @patch("src.ingestion.edgar_client.Company")
    def test_filing_metadata_extraction(self, mock_company_cls: MagicMock) -> None:
        """Verify all FilingMetadata fields are correctly populated."""
        mock_company = MagicMock()
        mock_filing = MagicMock()
        mock_filing.cik = "0000320193"
        mock_filing.company_name = "Apple Inc."
        mock_filing.filing_date = "2024-11-01"
        mock_filing.accession_number = "0000320193-24-000123"
        mock_filing.fiscal_year_end = "2024-09-28"
        mock_company.get_filings.return_value.filter.return_value = [mock_filing]
        mock_company_cls.return_value = mock_company

        client = EdgarClient(identity=self.IDENTITY)
        results = client.get_filings(ticker="AAPL", filing_type="10-K", num_filings=1)

        meta = results[0]
        assert meta.cik == "0000320193"
        assert meta.company_name == "Apple Inc."
        assert meta.ticker == "AAPL"
        assert meta.filing_type == "10-K"
        assert meta.accession_number == "0000320193-24-000123"
        assert meta.filing_date == date(2024, 11, 1)

    # -- download_filing ----------------------------------------------------

    @patch("src.ingestion.edgar_client.edgar.get_by_accession_number")
    def test_download_filing_returns_html(
        self,
        mock_get_by_accession: MagicMock,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """download_filing returns a non-empty HTML string."""
        mock_filing_obj = MagicMock()
        mock_filing_obj.html.return_value = "<html><body>10-K content</body></html>"
        mock_get_by_accession.return_value = mock_filing_obj

        client = EdgarClient(identity=self.IDENTITY)
        html = client.download_filing(sample_filing_metadata)

        assert isinstance(html, str)
        assert len(html) > 0
        assert "<html>" in html.lower()
        mock_get_by_accession.assert_called_once_with(
            sample_filing_metadata.accession_number
        )

    @patch("src.ingestion.edgar_client.Company")
    def test_get_filings_respects_num_filings(
        self, mock_company_cls: MagicMock
    ) -> None:
        """get_filings returns at most num_filings results."""
        mock_company = MagicMock()
        mock_filings = [MagicMock() for _ in range(10)]
        for i, mf in enumerate(mock_filings):
            mf.cik = "0000320193"
            mf.company_name = "Apple Inc."
            mf.filing_date = f"2024-01-{i+1:02d}"
            mf.accession_number = f"0000320193-24-{i:06d}"
        mock_company.get_filings.return_value.filter.return_value = mock_filings[:3]
        mock_company_cls.return_value = mock_company

        client = EdgarClient(identity=self.IDENTITY)
        results = client.get_filings(ticker="AAPL", filing_type="10-K", num_filings=3)

        assert len(results) <= 3

    @patch("src.ingestion.edgar_client.Company")
    def test_get_filings_default_filing_type(
        self, mock_company_cls: MagicMock
    ) -> None:
        """get_filings defaults to 10-K when filing_type is not specified."""
        mock_company = MagicMock()
        mock_company.get_filings.return_value.filter.return_value = []
        mock_company_cls.return_value = mock_company

        client = EdgarClient(identity=self.IDENTITY)
        client.get_filings(ticker="AAPL")

        # Verify filter was called with "10-K"
        mock_company.get_filings.return_value.filter.assert_called_once()
        call_args = mock_company.get_filings.return_value.filter.call_args
        assert "10-K" in str(call_args)


# ---------------------------------------------------------------------------
# FilingParser tests
# ---------------------------------------------------------------------------
class TestFilingParser:
    """Tests for FilingParser."""

    def test_parse_sections_from_10k(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """parse_sections identifies all four standard 10-K items."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        assert isinstance(sections, list)
        assert len(sections) >= 4

        section_names = [s.section_name for s in sections]
        # Expect these canonical section names (case-insensitive check)
        expected = ["Item 1", "Item 1A", "Item 7", "Item 8"]
        for exp in expected:
            assert any(
                exp.lower() in name.lower() for name in section_names
            ), f"Missing section: {exp}"

    def test_parse_sections_returns_filing_sections(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Each parsed section is a valid FilingSection with correct metadata."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        for section in sections:
            assert isinstance(section, FilingSection)
            assert section.metadata.ticker == "AAPL"
            assert section.metadata.filing_type == "10-K"
            assert len(section.content) > 0

    def test_extract_risk_factors(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Item 1A section contains risk-related content."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        risk_sections = [
            s for s in sections if "1a" in s.section_name.lower()
        ]
        assert len(risk_sections) == 1

        risk = risk_sections[0]
        assert "risk" in risk.content.lower() or "adverse" in risk.content.lower()
        assert len(risk.content) > 50

    def test_extract_mda(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Item 7 (MD&A) section contains discussion of financial results."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        mda_sections = [
            s for s in sections if "item 7" in s.section_name.lower()
        ]
        assert len(mda_sections) == 1

        mda = mda_sections[0]
        assert "revenue" in mda.content.lower() or "financial" in mda.content.lower()

    def test_parse_sections_content_not_empty(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Every parsed section must have non-empty content."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        for section in sections:
            assert section.content.strip(), (
                f"Section '{section.section_name}' has empty content"
            )

    # -- XBRL extraction ----------------------------------------------------

    def test_extract_xbrl_facts_returns_facts(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """extract_xbrl_facts returns a list of XBRLFact objects."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        assert isinstance(facts, list)
        assert len(facts) >= 4
        for fact in facts:
            assert isinstance(fact, XBRLFact)
            assert fact.metadata.ticker == "AAPL"

    def test_xbrl_facts_contain_key_concepts(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """XBRL extraction captures revenue, net income, assets, liabilities."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        concepts = [f.concept for f in facts]
        expected_concepts = [
            "us-gaap:Revenues",
            "us-gaap:NetIncomeLoss",
            "us-gaap:Assets",
            "us-gaap:Liabilities",
        ]
        for expected in expected_concepts:
            assert any(
                expected in c for c in concepts
            ), f"Missing XBRL concept: {expected}"

    def test_xbrl_facts_have_values(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Each XBRL fact has a non-empty value field."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        for fact in facts:
            assert fact.value, f"Fact {fact.concept} has no value"

    def test_xbrl_facts_have_units(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """XBRL facts include unit information (e.g., 'usd')."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        for fact in facts:
            assert fact.unit is not None, f"Fact {fact.concept} has no unit"

    def test_xbrl_to_dataframe(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """XBRL facts can be converted to a pandas DataFrame with expected columns."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        # Convert facts to DataFrame — this is the expected interface
        df = pd.DataFrame([fact.model_dump() for fact in facts])

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 4
        assert "concept" in df.columns
        assert "value" in df.columns
        assert "unit" in df.columns
        assert df["concept"].nunique() >= 4

    def test_xbrl_dataframe_numeric_values(
        self,
        sample_xbrl_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """XBRL fact values are parseable as numeric."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(sample_xbrl_html, sample_filing_metadata)

        for fact in facts:
            # Values should be numeric strings (potentially castable to float)
            try:
                float(fact.value)
            except ValueError:
                pytest.fail(f"Fact {fact.concept} has non-numeric value: {fact.value}")

    # -- Table preservation --------------------------------------------------

    def test_table_preservation(
        self,
        sample_html_with_table: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """Financial tables are preserved as markdown in the section's tables list."""
        parser = FilingParser()
        sections = parser.parse_sections(
            sample_html_with_table, sample_filing_metadata
        )

        # Find the section containing the table
        sections_with_tables = [s for s in sections if s.tables]
        assert len(sections_with_tables) >= 1, "No sections preserved tables"

        table_md = sections_with_tables[0].tables[0]
        # Table should be preserved as markdown with pipe separators
        assert "|" in table_md, "Table not in markdown pipe format"
        assert "Revenue" in table_md
        assert "Net Income" in table_md

    def test_table_not_split_across_sections(
        self,
        sample_filing_html: str,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """A table appearing in one section is not split across multiple sections."""
        parser = FilingParser()
        sections = parser.parse_sections(sample_filing_html, sample_filing_metadata)

        for section in sections:
            for table in section.tables:
                # Each table should have both header and data rows
                lines = [l for l in table.strip().split("\n") if l.strip()]
                if len(lines) > 1:
                    # A valid markdown table has at least header + separator + 1 row
                    assert len(lines) >= 3, (
                        f"Table in '{section.section_name}' appears incomplete "
                        f"(only {len(lines)} lines)"
                    )

    def test_parse_sections_empty_html(
        self,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """parse_sections returns an empty list for empty/minimal HTML."""
        parser = FilingParser()
        sections = parser.parse_sections("<html><body></body></html>", sample_filing_metadata)

        assert isinstance(sections, list)
        assert len(sections) == 0

    def test_extract_xbrl_facts_no_xbrl(
        self,
        sample_filing_metadata: FilingMetadata,
    ) -> None:
        """extract_xbrl_facts returns an empty list when there are no XBRL tags."""
        parser = FilingParser()
        facts = parser.extract_xbrl_facts(
            "<html><body><p>No XBRL here</p></body></html>",
            sample_filing_metadata,
        )

        assert isinstance(facts, list)
        assert len(facts) == 0


# ---------------------------------------------------------------------------
# Integration test — hits real SEC EDGAR API
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestIntegration:
    """Integration tests that hit the real SEC EDGAR API.

    Run with: pytest -m integration
    Skipped by default in CI; use -m integration to include.
    """

    @pytest.mark.slow
    def test_full_pipeline_apple_10k(self) -> None:
        """End-to-end: download AAPL 10-K, parse sections, extract XBRL."""
        client = EdgarClient(identity="FinRAGAgent test@example.com")

        # Step 1: Get filing metadata
        filings = client.get_filings(ticker="AAPL", filing_type="10-K", num_filings=1)
        assert len(filings) >= 1
        meta = filings[0]
        assert meta.ticker == "AAPL"
        assert meta.filing_type == "10-K"
        assert meta.cik  # Non-empty CIK
        assert meta.accession_number  # Non-empty accession number

        # Step 2: Download markdown (for sections) and HTML (for XBRL)
        markdown = client.download_filing_markdown(meta)
        assert isinstance(markdown, str)
        assert len(markdown) > 1000

        html = client.download_filing(meta)
        assert isinstance(html, str)
        assert len(html) > 1000  # A real 10-K is large

        # Step 3: Parse sections from markdown (matches ingest.py pipeline)
        parser = FilingParser()
        sections = parser.parse_sections(markdown, meta)
        assert len(sections) >= 3  # At least a few standard sections

        section_names_lower = [s.section_name.lower() for s in sections]
        # 10-K must have at least risk factors and MD&A
        assert any("1a" in n for n in section_names_lower), "Missing Item 1A"
        assert any("7" in n for n in section_names_lower), "Missing Item 7"

        for section in sections:
            assert isinstance(section, FilingSection)
            assert len(section.content) > 0

        # Step 4: Extract XBRL facts from HTML
        facts = parser.extract_xbrl_facts(html, meta)
        assert isinstance(facts, list)
        # Real filings should have many XBRL facts
        assert len(facts) >= 5

        # Verify we get key financial concepts
        concepts = {f.concept for f in facts}
        # At least one revenue-related and one asset-related concept
        assert any("revenue" in c.lower() for c in concepts), (
            "No revenue concept found in XBRL"
        )

        # Step 5: Convert to DataFrame
        df = pd.DataFrame([f.model_dump() for f in facts])
        assert len(df) >= 5
        assert "concept" in df.columns
        assert "value" in df.columns
