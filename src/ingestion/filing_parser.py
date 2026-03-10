"""Parse 10-K and 10-Q filings into structured sections and XBRL facts.

Extracts named sections (Item 1, Item 1A, Item 7, etc.) from raw
filing HTML or markdown and parses XBRL inline data into structured facts.
"""

import re

from src.ingestion.models import FilingMetadata, FilingSection, XBRLFact

# Regex to match Item headers inside heading tags (h1–h4) — unit test fixtures
_ITEM_HTML_RE = re.compile(
    r"<h[1-4][^>]*>\s*(ITEM\s+\d+[A-Z]?\.?[^<]*?)\s*</h[1-4]>",
    re.IGNORECASE | re.DOTALL,
)

# Unified regex: finds "ITEM N. <title>" lines that aren't in pipe-table rows.
# Works on edgartools markdown (## headers, <div>-wrapped, or bare text).
_ITEM_UNIFIED_RE = re.compile(
    r"^[^|\n]*?(ITEM\s+\d+[A-Z]?\.\s+\S[^\n]{5,})",
    re.IGNORECASE | re.MULTILINE,
)

# Keywords that indicate a reference to an Item, not an actual section header
_REFERENCE_KEYWORDS = ("refer to", "part i,", "part ii,", "this form", "sections:")

# Regex for inline XBRL nonfraction tags
_XBRL_RE = re.compile(
    r"<ix:nonfraction\s+([^>]*)>([^<]*)</ix:nonfraction>",
    re.IGNORECASE,
)

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

# Markdown table: consecutive lines starting with |
_MD_TABLE_RE = re.compile(
    r"((?:^[ \t]*\|[^\n]+\|[ \t]*\n){2,})",
    re.MULTILINE,
)


class FilingParser:
    """Parser for SEC 10-K and 10-Q filing documents."""

    def parse_sections(
        self,
        raw_html: str,
        metadata: FilingMetadata,
    ) -> list[FilingSection]:
        """Parse a raw filing into named sections.

        Supports HTML (with <h1-4> Item headers) and markdown/mixed formats
        (## headers, <div>-wrapped headers, or bare text).

        Args:
            raw_html: Raw HTML or markdown content of the filing.
            metadata: Metadata for the filing being parsed.

        Returns:
            List of FilingSection objects, one per identified section.
        """
        # HTML with <h1-4> tags (unit test fixtures)
        html_matches = list(_ITEM_HTML_RE.finditer(raw_html))
        if html_matches:
            return self._parse_html_sections(raw_html, html_matches, metadata)

        # Markdown / mixed format from edgartools
        return self._parse_unified_sections(raw_html, metadata)

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
        facts: list[XBRLFact] = []

        for match in _XBRL_RE.finditer(raw_html):
            attrs = dict(_ATTR_RE.findall(match.group(1)))
            value = match.group(2).strip()

            concept = attrs.get("name", "")
            if not concept or not value:
                continue

            facts.append(
                XBRLFact(
                    concept=concept,
                    value=value,
                    unit=attrs.get("unitRef", attrs.get("unitref")),
                    metadata=metadata,
                )
            )

        return facts

    # ------------------------------------------------------------------
    # HTML section parsing (unit test fixtures)
    # ------------------------------------------------------------------

    def _parse_html_sections(
        self,
        raw_html: str,
        matches: list[re.Match],
        metadata: FilingMetadata,
    ) -> list[FilingSection]:
        """Parse sections from HTML with <h1-4> Item headers."""
        sections: list[FilingSection] = []
        for idx, match in enumerate(matches):
            section_name = re.sub(r"\s+", " ", match.group(1)).strip()

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_html)
            raw_content = raw_html[start:end]

            tables = self._extract_html_tables(raw_content)
            content = self._strip_html(raw_content)

            if content:
                sections.append(
                    FilingSection(
                        section_name=section_name,
                        content=content,
                        metadata=metadata,
                        tables=tables,
                    )
                )

        return sections

    # ------------------------------------------------------------------
    # Unified section parsing (real filings — markdown/mixed formats)
    # ------------------------------------------------------------------

    def _parse_unified_sections(
        self,
        text: str,
        metadata: FilingMetadata,
    ) -> list[FilingSection]:
        """Parse sections using unified ITEM regex across all formats.

        Finds all 'ITEM N. <title>' patterns, deduplicates by Item number
        (keeping the first occurrence), skips inline references, and
        extracts content between successive headers.
        """
        # Collect first occurrence of each Item number
        seen_items: dict[str, tuple[int, int, str]] = {}  # item_num -> (start, end, name)

        for match in _ITEM_UNIFIED_RE.finditer(text):
            raw_name = match.group(1).strip()
            # Clean trailing </div> or other HTML tags
            clean_name = re.sub(r"\s*<[^>]+>\s*$", "", raw_name).strip()

            item_num_match = re.match(r"(?i)ITEM\s+(\d+[A-Z]?)", clean_name)
            if not item_num_match:
                continue
            item_num = item_num_match.group(1).upper()

            # Skip inline references like "refer to ... Item 7"
            context_start = max(0, match.start() - 80)
            context = text[context_start:match.start()].lower()
            if any(kw in context for kw in _REFERENCE_KEYWORDS):
                continue

            if item_num not in seen_items:
                seen_items[item_num] = (match.start(), match.end(), clean_name)

        if not seen_items:
            return []

        # Sort by position in the document
        ordered = sorted(seen_items.values(), key=lambda t: t[0])

        sections: list[FilingSection] = []
        for idx, (start, end, section_name) in enumerate(ordered):
            content_start = end
            content_end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(text)
            raw_content = text[content_start:content_end]

            tables = self._extract_md_tables(raw_content)
            content = self._strip_md_content(raw_content)

            if content:
                sections.append(
                    FilingSection(
                        section_name=section_name,
                        content=content,
                        metadata=metadata,
                        tables=tables,
                    )
                )

        return sections

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def _extract_html_tables(self, html: str) -> list[str]:
        """Extract HTML tables and convert to markdown format."""
        tables: list[str] = []
        for match in _TABLE_RE.finditer(html):
            md = self._html_table_to_markdown(match.group(1))
            if md:
                tables.append(md)
        return tables

    @staticmethod
    def _extract_md_tables(text: str) -> list[str]:
        """Extract markdown tables (pipe-delimited) from text."""
        tables: list[str] = []
        for match in _MD_TABLE_RE.finditer(text):
            table = match.group(1).strip()
            lines = [ln for ln in table.split("\n") if ln.strip()]
            if len(lines) >= 3:
                tables.append(table)
        return tables

    @staticmethod
    def _html_table_to_markdown(table_inner_html: str) -> str:
        """Convert the inner HTML of a <table> to a markdown table."""
        rows: list[list[str]] = []
        for row_match in _ROW_RE.finditer(table_inner_html):
            cells = [
                _TAG_RE.sub("", c.group(1)).strip()
                for c in _CELL_RE.finditer(row_match.group(1))
            ]
            if cells:
                rows.append(cells)

        if len(rows) < 2:
            return ""

        header = rows[0]
        ncols = len(header)
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        for row in rows[1:]:
            padded = (row + [""] * ncols)[:ncols]
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Text cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags and collapse whitespace."""
        text = _TAG_RE.sub(" ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _strip_md_content(text: str) -> str:
        """Clean markdown content: remove table blocks, collapse whitespace."""
        # Remove markdown table blocks to avoid double-counting in content
        cleaned = _MD_TABLE_RE.sub("", text)
        # Remove sub-headers that aren't Item headers
        cleaned = re.sub(r"^#{1,6}\s+(?!Item\s)", "", cleaned, flags=re.MULTILINE)
        # Remove remaining HTML tags (div wrappers, etc.)
        cleaned = _TAG_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
