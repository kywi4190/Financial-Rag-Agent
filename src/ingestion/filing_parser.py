"""Parse 10-K and 10-Q filings into structured sections and XBRL facts.

Extracts named sections (Item 1, Item 1A, Item 7, etc.) from raw
filing HTML and parses XBRL inline data into structured facts.
"""

from src.ingestion.models import FilingMetadata, FilingSection, XBRLFact


class FilingParser:
    """Parser for SEC 10-K and 10-Q filing documents."""

    def parse_sections(
        self,
        raw_html: str,
        metadata: FilingMetadata,
    ) -> list[FilingSection]:
        """Parse a raw filing into named sections.

        Args:
            raw_html: Raw HTML content of the filing.
            metadata: Metadata for the filing being parsed.

        Returns:
            List of FilingSection objects, one per identified section.
        """
        ...

    def extract_xbrl_facts(
        self,
        raw_html: str,
        metadata: FilingMetadata,
    ) -> list[XBRLFact]:
        """Extract XBRL financial facts from inline XBRL markup.

        Args:
            raw_html: Raw HTML content containing inline XBRL tags.
            metadata: Metadata for the filing being parsed.

        Returns:
            List of XBRLFact objects.
        """
        ...
