"""Financial table extraction and formatting.

Detects tables in filing HTML, converts them to structured markdown,
and classifies whether a table contains financial data.
"""

import re

from bs4 import BeautifulSoup


def detect_tables(html_content: str) -> list[str]:
    """Find all HTML tables in filing content.

    Args:
        html_content: Raw HTML content potentially containing tables.

    Returns:
        List of individual table HTML strings.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    return [str(table) for table in soup.find_all("table")]


def html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to clean markdown format.

    Preserves exact numbers, column alignment, and row/column headers.

    Args:
        html: A single HTML table string.

    Returns:
        Markdown-formatted table string.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html.strip()

    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["td", "th"]):
            text = cell.get_text(strip=True)
            text = re.sub(r"\s+", " ", text)
            cells.append(text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    lines: list[str] = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def is_financial_table(table_md: str) -> bool:
    """Classify whether a markdown table contains financial data.

    Uses heuristic pattern matching to distinguish financial data tables
    (income statements, balance sheets, etc.) from decorative/navigation tables.

    Args:
        table_md: Markdown-formatted table string.

    Returns:
        True if the table appears to contain financial data.
    """
    financial_patterns = [
        r"\$[\d,]+",
        r"\d{1,3}(,\d{3})+",
        r"revenue|income|loss|assets|liabilities|equity|earnings",
        r"operating|net\s+income|gross\s+profit|total",
        r"balance\s+sheet|cash\s+flow|statement",
        r"\(\d[\d,]*\)",
    ]

    text_lower = table_md.lower()
    matches = sum(
        1 for pattern in financial_patterns if re.search(pattern, text_lower)
    )
    return matches >= 2
