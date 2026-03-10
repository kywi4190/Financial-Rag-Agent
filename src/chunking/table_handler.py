"""Financial table extraction and formatting.

Detects tables in filing HTML, converts them to a structured markdown
format suitable for embedding, and preserves numeric precision.
"""

from src.chunking.models import DocumentChunk
from src.ingestion.models import FilingMetadata


class TableHandler:
    """Extracts and formats financial tables from filing HTML."""

    def extract_tables(self, html: str) -> list[str]:
        """Extract HTML tables from raw filing content.

        Args:
            html: Raw HTML content potentially containing tables.

        Returns:
            List of individual table HTML strings.
        """
        ...

    def table_to_markdown(self, table_html: str) -> str:
        """Convert an HTML table to clean markdown format.

        Args:
            table_html: A single HTML table string.

        Returns:
            Markdown-formatted table string.
        """
        ...

    def create_table_chunk(
        self,
        table_markdown: str,
        metadata: FilingMetadata,
        section_name: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """Create a DocumentChunk from a formatted table.

        Args:
            table_markdown: Markdown-formatted table content.
            metadata: Source filing metadata.
            section_name: Section the table was found in.
            chunk_index: Position index for the chunk.

        Returns:
            A DocumentChunk with is_table=True in its metadata.
        """
        ...
