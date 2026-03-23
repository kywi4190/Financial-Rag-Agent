"""Pydantic models for agent outputs.

Defines structured representations of agent responses including
citations, financial metrics, comparisons, and investment memos.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source citation for a generated claim.

    Attributes:
        source_document: Name or identifier of the source filing.
        section: Filing section (e.g., 'Item 7. MD&A').
        ticker: Company stock ticker.
        year: Fiscal year of the filing.
        quote_snippet: Short verbatim excerpt (max 100 chars).
    """

    source_document: str
    section: str
    ticker: str
    year: int
    quote_snippet: str = Field(max_length=100)


class AnswerWithCitations(BaseModel):
    """Structured answer with source citations.

    Attributes:
        answer: Generated answer text.
        citations: Source citations supporting the answer.
        confidence: Confidence score between 0 and 1.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    contexts_used: list[dict] = Field(default_factory=list)


class FinancialMetric(BaseModel):
    """A single financial metric with its source.

    Attributes:
        name: Metric name (e.g., 'Revenue', 'Gross Margin').
        value: Numeric value as Decimal for precision.
        unit: Unit of measurement (e.g., 'USD', '%').
        period: Reporting period (e.g., 'FY2024').
        source: Citation for where this metric was found.
    """

    name: str
    value: Decimal
    unit: str
    period: str
    source: Citation


class CompanyComparison(BaseModel):
    """Comparison of a metric across multiple companies.

    Attributes:
        metric_name: Name of the metric being compared.
        companies: Map of ticker to list of metrics across periods.
        analysis: Textual analysis of the comparison.
    """

    metric_name: str
    companies: dict[str, list[FinancialMetric]]
    analysis: str


class MemoSection(BaseModel):
    """A single section of an investment memo.

    Attributes:
        title: Section heading.
        content: Section body text.
        citations: Source citations for this section.
    """

    title: str
    content: str
    citations: list[Citation] = Field(default_factory=list)


class InvestmentMemo(BaseModel):
    """Structured investment memo output.

    Attributes:
        ticker: Stock ticker symbol.
        company_name: Legal company name.
        date_generated: When the memo was produced.
        executive_summary: High-level investment thesis.
        company_overview: Business description and market position.
        financial_highlights: Key financial metrics and trends.
        risk_factors: Material risks identified from filings.
        mda_synthesis: Management Discussion & Analysis summary.
        bull_bear_case: Bull and bear investment scenarios.
    """

    ticker: str
    company_name: str
    date_generated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    executive_summary: MemoSection
    company_overview: MemoSection
    financial_highlights: MemoSection
    risk_factors: MemoSection
    mda_synthesis: MemoSection
    bull_bear_case: MemoSection

    def to_markdown(self) -> str:
        """Render the memo as a formatted Markdown document.

        Returns:
            Complete memo as a Markdown string.
        """
        sections = [
            self.executive_summary,
            self.company_overview,
            self.financial_highlights,
            self.risk_factors,
            self.mda_synthesis,
            self.bull_bear_case,
        ]

        lines: list[str] = [
            f"# Investment Memo: {self.ticker} — {self.company_name}",
            f"*Generated: {self.date_generated.strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
        ]

        for section in sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
            if section.citations:
                lines.append("### Sources")
                for c in section.citations:
                    lines.append(
                        f"- {c.source_document}, {c.section} "
                        f"(*\"{c.quote_snippet}\"*)"
                    )
                lines.append("")

        return "\n".join(lines)


# Backwards-compatible aliases used by memo_generator
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FinancialCalculation(BaseModel):
    """Result of a financial calculation tool.

    Attributes:
        metric_name: Name of the calculated metric.
        value: Computed numeric value.
        unit: Unit of the result (e.g., '%', 'USD').
        inputs: Dictionary of input values used.
        formula: Description of the formula applied.
    """

    metric_name: str
    value: float
    unit: str = ""
    inputs: dict[str, float] = Field(default_factory=dict)
    formula: str = ""
