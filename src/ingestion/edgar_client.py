"""SEC EDGAR download client using edgartools.

Handles downloading 10-K and 10-Q filings from SEC EDGAR,
respecting rate limits and the required user-agent identity.
"""

from src.ingestion.models import FilingMetadata


class EdgarClient:
    """Client for downloading SEC filings via the EDGAR API.

    Args:
        identity: SEC EDGAR user-agent identity string.
    """

    def __init__(self, identity: str) -> None:
        """Initialize the EDGAR client with the given identity."""
        ...

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
        """
        ...

    def download_filing(self, metadata: FilingMetadata) -> str:
        """Download the full text of a filing.

        Args:
            metadata: Filing metadata identifying the document.

        Returns:
            Raw HTML/text content of the filing.
        """
        ...
