"""Pydantic models for agent outputs.

Defines structured representations of agent responses including
query answers, financial calculations, and investment memos.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.retrieval.models import SearchResult


class SourceAttribution(BaseModel):
    """Attribution to a specific source chunk.

    Attributes:
        chunk_id: ID of the source chunk.
        text_excerpt: Relevant excerpt from the source.
        filing_type: Type of the source filing (e.g., '10-K').
        company: Company name from the source.
        section: Filing section name.
    """

    chunk_id: str
    text_excerpt: str
    filing_type: str
    company: str
    section: str


class QueryResponse(BaseModel):
    """Structured response from the query engine.

    Attributes:
        query: The original user query.
        answer: Generated answer text.
        sources: List of source attributions supporting the answer.
        confidence: Confidence score between 0 and 1.
        timestamp: When the response was generated.
    """

    query: str
    answer: str
    sources: list[SourceAttribution] = Field(default_factory=list)
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FinancialCalculation(BaseModel):
    """Result of a financial calculation tool.

    Attributes:
        metric_name: Name of the calculated metric (e.g., 'gross_margin').
        value: Computed numeric value.
        unit: Unit of the result (e.g., '%', 'USD').
        inputs: Dictionary of input values used in the calculation.
        formula: Description of the formula applied.
    """

    metric_name: str
    value: float
    unit: str = ""
    inputs: dict[str, float] = Field(default_factory=dict)
    formula: str = ""


class InvestmentMemo(BaseModel):
    """Structured investment memo output.

    Attributes:
        company: Company name.
        ticker: Stock ticker symbol.
        summary: Executive summary paragraph.
        strengths: List of identified strengths.
        risks: List of identified risks.
        financial_highlights: Key financial metrics and observations.
        recommendation: Investment recommendation text.
        sources: Source attributions used to generate the memo.
    """

    company: str
    ticker: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    financial_highlights: list[str] = Field(default_factory=list)
    recommendation: str = ""
    sources: list[SourceAttribution] = Field(default_factory=list)
