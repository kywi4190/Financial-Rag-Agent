"""Pydantic models for SEC filing data.

Defines the structured representations of filings, filing sections,
and extracted XBRL financial facts.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class FilingMetadata(BaseModel):
    """Metadata for a single SEC filing.

    Attributes:
        cik: Central Index Key for the filer.
        company_name: Legal name of the filing company.
        ticker: Stock ticker symbol.
        filing_type: SEC form type (e.g., '10-K', '10-Q').
        filing_date: Date the filing was submitted.
        accession_number: Unique SEC accession number.
        fiscal_year_end: End date of the fiscal year covered.
    """

    cik: str
    company_name: str
    ticker: str
    filing_type: str
    filing_date: date
    accession_number: str
    fiscal_year_end: Optional[date] = None


class FilingSection(BaseModel):
    """A parsed section from an SEC filing.

    Attributes:
        section_name: Canonical section name (e.g., 'Item 1A. Risk Factors').
        content: Raw text content of the section.
        metadata: Parent filing metadata.
        tables: List of raw HTML or markdown tables found in the section.
    """

    section_name: str
    content: str
    metadata: FilingMetadata
    tables: list[str] = Field(default_factory=list)


class XBRLFact(BaseModel):
    """A single XBRL financial fact extracted from a filing.

    Attributes:
        concept: XBRL taxonomy concept (e.g., 'us-gaap:Revenue').
        value: Numeric or string value of the fact.
        unit: Unit of measurement (e.g., 'USD', 'shares').
        period_start: Start date of the reporting period.
        period_end: End date of the reporting period.
        metadata: Parent filing metadata.
    """

    concept: str
    value: str
    unit: Optional[str] = None
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    metadata: FilingMetadata
