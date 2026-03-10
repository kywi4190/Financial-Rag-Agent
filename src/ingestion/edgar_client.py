"""SEC EDGAR download client using edgartools.

Handles downloading 10-K and 10-Q filings from SEC EDGAR,
respecting rate limits and the required user-agent identity.
"""

import logging
from datetime import date
from typing import Optional

import edgar
from edgar import Company

from src.ingestion.models import FilingMetadata

logger = logging.getLogger(__name__)


class EdgarClient:
    """Client for downloading SEC filings via the EDGAR API.

    Args:
        identity: SEC EDGAR user-agent identity string.
    """

    def __init__(self, identity: str) -> None:
        """Initialize the EDGAR client with the given identity."""
        self.identity = identity
        edgar.set_identity(identity)

    def get_filings(
        self,
        ticker: str,
        filing_type: str = "10-K",
        num_filings: int = 5,
    ) -> list[FilingMetadata]:
        """Retrieve filing metadata for a company.

        Args:
            ticker: Stock ticker symbol.
            filing_type: SEC form type to retrieve.
            num_filings: Maximum number of filings to return.

        Returns:
            List of FilingMetadata objects.

        Raises:
            ValueError: If the ticker is not found.
        """
        company = Company(ticker)
        filings_raw = company.get_filings().filter(form=filing_type)

        results: list[FilingMetadata] = []
        for i, filing in enumerate(filings_raw):
            if i >= num_filings:
                break
            results.append(self._to_metadata(filing, ticker, filing_type))
        return results

    def download_filing(self, metadata: FilingMetadata) -> str:
        """Download the full HTML text of a filing.

        Args:
            metadata: Filing metadata identifying the document.

        Returns:
            Raw HTML content of the filing.
        """
        filing = self._get_filing_obj(metadata)
        return filing.html()

    def download_filing_markdown(self, metadata: FilingMetadata) -> str:
        """Download the filing as structured markdown.

        Args:
            metadata: Filing metadata identifying the document.

        Returns:
            Markdown content of the filing with ## Item headers.
        """
        filing = self._get_filing_obj(metadata)
        return filing.markdown()

    def _get_filing_obj(self, metadata: FilingMetadata) -> object:
        """Retrieve the raw edgartools filing object."""
        return edgar.get_by_accession_number(metadata.accession_number)

    def _to_metadata(
        self,
        filing: object,
        ticker: str,
        filing_type: str,
    ) -> FilingMetadata:
        """Convert an edgartools filing object or dict to FilingMetadata."""
        if isinstance(filing, dict):
            return FilingMetadata(
                cik=filing.get("cik", ""),
                company_name=filing.get("company_name", ""),
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing.get("filing_date"),
                accession_number=filing.get("accession_number", ""),
                fiscal_year_end=filing.get("fiscal_year_end"),
            )

        # Handle edgartools Filing objects (and MagicMock in tests)
        # Real edgartools uses .company; test mocks set .company_name
        company_name = getattr(filing, "company_name", None)
        if not isinstance(company_name, str):
            company_name = str(getattr(filing, "company", ""))

        fiscal_year_end: Optional[date] = None
        raw_fye = getattr(filing, "fiscal_year_end", None)
        if isinstance(raw_fye, (str, date)):
            fiscal_year_end = raw_fye  # type: ignore[assignment]

        # Real edgartools returns period_of_report for fiscal year end
        if fiscal_year_end is None:
            raw_por = getattr(filing, "period_of_report", None)
            if isinstance(raw_por, (str, date)):
                fiscal_year_end = raw_por  # type: ignore[assignment]

        return FilingMetadata(
            cik=str(filing.cik).zfill(10),  # type: ignore[attr-defined]
            company_name=company_name,
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing.filing_date,  # type: ignore[attr-defined]
            accession_number=str(filing.accession_number),  # type: ignore[attr-defined]
            fiscal_year_end=fiscal_year_end,
        )
