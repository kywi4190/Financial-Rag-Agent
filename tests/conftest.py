"""Shared pytest fixtures for the Financial RAG Agent test suite."""

from datetime import date

import pytest

from src.ingestion.models import FilingMetadata, FilingSection, XBRLFact


@pytest.fixture
def sample_filing_metadata() -> FilingMetadata:
    """Create a sample AAPL 10-K FilingMetadata for use in tests."""
    return FilingMetadata(
        cik="0000320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        filing_type="10-K",
        filing_date=date(2024, 11, 1),
        accession_number="0000320193-24-000123",
        fiscal_year_end=date(2024, 9, 28),
    )


@pytest.fixture
def sample_10q_metadata() -> FilingMetadata:
    """Create a sample AAPL 10-Q FilingMetadata."""
    return FilingMetadata(
        cik="0000320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        filing_type="10-Q",
        filing_date=date(2024, 8, 2),
        accession_number="0000320193-24-000099",
        fiscal_year_end=None,
    )


@pytest.fixture
def sample_filing_html() -> str:
    """Return a minimal sample 10-K HTML string for parsing tests.

    Contains realistic section headers for Item 1, 1A, 7, and 8,
    plus inline XBRL tags and a markdown-style financial table.
    """
    return """
<html>
<body>
<div>
<h2>ITEM 1. BUSINESS</h2>
<p>Apple Inc. designs, manufactures, and markets smartphones,
personal computers, tablets, wearables, and accessories worldwide.
The Company also sells various related services.</p>

<h2>ITEM 1A. RISK FACTORS</h2>
<p>The Company's business, reputation, results of operations, financial
condition, and stock price can be affected by a number of factors,
whether currently known or unknown, including those described below.</p>
<p>Global market conditions could materially adversely affect the Company.
The Company has international operations with sales in many countries.
Currency fluctuations could negatively impact reported earnings.</p>

<h2>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION
AND RESULTS OF OPERATIONS</h2>
<p>The following discussion should be read in conjunction with the
consolidated financial statements.</p>
<p>Total net revenue was $383.3 billion for fiscal year 2024, compared
to $394.3 billion for fiscal year 2023.</p>

<table>
<tr><th>Metric</th><th>2024</th><th>2023</th></tr>
<tr><td>Revenue</td><td>$383,285</td><td>$394,328</td></tr>
<tr><td>Net Income</td><td>$93,736</td><td>$96,995</td></tr>
</table>

<h2>ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA</h2>
<p>See the consolidated financial statements and notes thereto.</p>

<ix:nonfraction name="us-gaap:Revenues"
    contextRef="FY2024" unitRef="usd" decimals="-6">383285000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:NetIncomeLoss"
    contextRef="FY2024" unitRef="usd" decimals="-6">93736000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:Assets"
    contextRef="FY2024" unitRef="usd" decimals="-6">352583000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:Liabilities"
    contextRef="FY2024" unitRef="usd" decimals="-6">290437000000</ix:nonfraction>
</div>
</body>
</html>
"""


@pytest.fixture
def sample_xbrl_html() -> str:
    """Return HTML with inline XBRL tags only, for focused XBRL tests."""
    return """
<html><body>
<ix:nonfraction name="us-gaap:Revenues"
    contextRef="FY2024" unitRef="usd" decimals="-6">383285000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:NetIncomeLoss"
    contextRef="FY2024" unitRef="usd" decimals="-6">93736000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:Assets"
    contextRef="FY2024" unitRef="usd" decimals="-6">352583000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:Liabilities"
    contextRef="FY2024" unitRef="usd" decimals="-6">290437000000</ix:nonfraction>
<ix:nonfraction name="us-gaap:StockholdersEquity"
    contextRef="FY2024" unitRef="usd" decimals="-6">62146000000</ix:nonfraction>
</body></html>
"""


@pytest.fixture
def sample_html_with_table() -> str:
    """Return HTML containing a financial table to test table preservation."""
    return """
<html><body>
<h2>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h2>
<p>Summary of operating results:</p>
<table>
<tr><th>Metric</th><th>2024</th><th>2023</th><th>2022</th></tr>
<tr><td>Total Revenue</td><td>$383,285</td><td>$394,328</td><td>$365,817</td></tr>
<tr><td>Cost of Sales</td><td>$210,352</td><td>$214,137</td><td>$201,471</td></tr>
<tr><td>Gross Margin</td><td>$172,933</td><td>$180,191</td><td>$164,346</td></tr>
<tr><td>Net Income</td><td>$93,736</td><td>$96,995</td><td>$94,680</td></tr>
</table>
</body></html>
"""


@pytest.fixture
def mock_edgar_filings_response() -> list[dict]:
    """Return a list of dicts mimicking edgartools filing results."""
    return [
        {
            "cik": "0000320193",
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-11-01",
            "accession_number": "0000320193-24-000123",
        },
        {
            "cik": "0000320193",
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2023-11-03",
            "accession_number": "0000320193-23-000106",
        },
    ]


@pytest.fixture
def sample_chunks() -> list:
    """Return a list of sample DocumentChunk objects for retrieval tests."""
    ...
