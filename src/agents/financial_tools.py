"""Agent tools for XBRL lookup, financial calculations, and retrieval.

Provides tool functions that can be bound to a LlamaIndex agent
for structured financial data retrieval and ratio calculations.
"""

import logging
from decimal import Decimal
from typing import Any

import pandas as pd

from src.agents.models import (
    Citation,
    CompanyComparison,
    FinancialCalculation,
    FinancialMetric,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XBRL data store — populated by the ingestion pipeline
# ---------------------------------------------------------------------------

# Maps (ticker, year) -> DataFrame with columns: concept, value, unit, period
_xbrl_store: dict[tuple[str, int], pd.DataFrame] = {}


def register_xbrl_data(ticker: str, year: int, df: pd.DataFrame) -> None:
    """Register a DataFrame of XBRL facts for a company/year.

    Args:
        ticker: Stock ticker symbol.
        year: Fiscal year.
        df: DataFrame with columns [concept, value, unit, period].
    """
    _xbrl_store[(ticker.upper(), year)] = df
    logger.info("Registered XBRL data for %s %d (%d rows)", ticker, year, len(df))


def get_xbrl_store() -> dict[tuple[str, int], pd.DataFrame]:
    """Return a reference to the XBRL data store (for testing)."""
    return _xbrl_store


# ---------------------------------------------------------------------------
# Tool: XBRL Lookup
# ---------------------------------------------------------------------------


def xbrl_lookup_tool(
    ticker: str,
    concept: str,
    period: str | None = None,
) -> dict[str, Any]:
    """Query XBRL data for a specific metric from stored DataFrames.

    Args:
        ticker: Stock ticker symbol.
        concept: XBRL concept or common name (e.g., 'Revenue',
                 'us-gaap:Revenues').
        period: Optional period filter (e.g., 'FY2024').

    Returns:
        Dictionary with matched facts or an error message.
    """
    ticker_upper = ticker.upper()
    concept_lower = concept.lower()

    matching_frames: list[tuple[str, int, pd.DataFrame]] = []
    for (tk, yr), df in _xbrl_store.items():
        if tk == ticker_upper:
            matching_frames.append((tk, yr, df))

    if not matching_frames:
        return {"error": f"No XBRL data found for ticker {ticker_upper}"}

    results: list[dict[str, Any]] = []
    for tk, yr, df in matching_frames:
        mask = df["concept"].str.lower().str.contains(concept_lower, na=False)
        if period:
            period_mask = df["period"].str.contains(period, case=False, na=False)
            mask = mask & period_mask
        matched = df[mask]
        for _, row in matched.iterrows():
            results.append({
                "ticker": tk,
                "year": yr,
                "concept": row["concept"],
                "value": row["value"],
                "unit": row.get("unit", "USD"),
                "period": row.get("period", f"FY{yr}"),
            })

    if not results:
        return {"error": f"Concept '{concept}' not found for {ticker_upper}"}

    logger.info("XBRL lookup: %s/%s returned %d results", ticker, concept, len(results))
    return {"ticker": ticker_upper, "concept": concept, "results": results}


# ---------------------------------------------------------------------------
# Tool: Calculate Ratio
# ---------------------------------------------------------------------------

_RATIO_FORMULAS: dict[str, tuple[str, str, str]] = {
    "debt_to_equity": ("Liabilities", "StockholdersEquity", "x"),
    "current_ratio": ("AssetsCurrent", "LiabilitiesCurrent", "x"),
    "gross_margin": ("GrossProfit", "Revenues", "%"),
    "net_margin": ("NetIncomeLoss", "Revenues", "%"),
    "return_on_assets": ("NetIncomeLoss", "Assets", "%"),
    "return_on_equity": ("NetIncomeLoss", "StockholdersEquity", "%"),
}


def calculate_ratio_tool(
    ticker: str,
    ratio_name: str,
    year: int | None = None,
    numerator: float | None = None,
    denominator: float | None = None,
) -> FinancialCalculation:
    """Compute a financial ratio from XBRL data or provided values.

    If numerator/denominator are not supplied, looks up the values
    from the XBRL store using the named ratio's known formula.

    Args:
        ticker: Stock ticker symbol.
        ratio_name: Name of the ratio (e.g., 'debt_to_equity', 'gross_margin').
        year: Fiscal year for XBRL lookup (required if values not provided).
        numerator: Optional explicit numerator value.
        denominator: Optional explicit denominator value.

    Returns:
        FinancialCalculation with the computed ratio.
    """
    if numerator is not None and denominator is not None:
        if denominator == 0:
            return FinancialCalculation(
                metric_name=ratio_name,
                value=0.0,
                unit="",
                inputs={"numerator": numerator, "denominator": denominator},
                formula=f"{ratio_name} = numerator / denominator (division by zero)",
            )
        value = numerator / denominator
        unit = "%" if ratio_name in _RATIO_FORMULAS and _RATIO_FORMULAS[ratio_name][2] == "%" else "x"
        if unit == "%":
            value *= 100
        return FinancialCalculation(
            metric_name=ratio_name,
            value=round(value, 4),
            unit=unit,
            inputs={"numerator": numerator, "denominator": denominator},
            formula=f"{ratio_name} = {numerator} / {denominator}",
        )

    if ratio_name not in _RATIO_FORMULAS:
        return FinancialCalculation(
            metric_name=ratio_name,
            value=0.0,
            formula=f"Unknown ratio: {ratio_name}. Known: {list(_RATIO_FORMULAS.keys())}",
        )

    num_concept, den_concept, unit = _RATIO_FORMULAS[ratio_name]
    ticker_upper = ticker.upper()

    num_result = xbrl_lookup_tool(ticker_upper, num_concept)
    den_result = xbrl_lookup_tool(ticker_upper, den_concept)

    if "error" in num_result or "error" in den_result:
        return FinancialCalculation(
            metric_name=ratio_name,
            value=0.0,
            formula=f"Missing data: num={num_result}, den={den_result}",
        )

    # Pick the matching year or the first result
    def _pick(results: list[dict], yr: int | None) -> float:
        for r in results:
            if yr is None or r["year"] == yr:
                return float(r["value"])
        return float(results[0]["value"])

    num_val = _pick(num_result["results"], year)
    den_val = _pick(den_result["results"], year)

    if den_val == 0:
        return FinancialCalculation(
            metric_name=ratio_name, value=0.0, unit=unit,
            inputs={"numerator": num_val, "denominator": den_val},
            formula=f"{ratio_name} = {num_concept} / {den_concept} (division by zero)",
        )

    value = num_val / den_val
    if unit == "%":
        value *= 100

    return FinancialCalculation(
        metric_name=ratio_name,
        value=round(value, 4),
        unit=unit,
        inputs={"numerator": num_val, "denominator": den_val},
        formula=f"{ratio_name} = {num_concept} / {den_concept}",
    )


# ---------------------------------------------------------------------------
# Tool: Compare Metrics
# ---------------------------------------------------------------------------


def compare_metrics_tool(
    metric_concept: str,
    tickers: list[str],
    years: list[int] | None = None,
) -> CompanyComparison:
    """Compare a metric across multiple tickers and/or years.

    Args:
        metric_concept: XBRL concept or common name to compare.
        tickers: List of stock ticker symbols.
        years: Optional list of years to include.

    Returns:
        CompanyComparison with metrics per company and analysis text.
    """
    companies: dict[str, list[FinancialMetric]] = {}

    for ticker in tickers:
        result = xbrl_lookup_tool(ticker, metric_concept)
        metrics: list[FinancialMetric] = []
        if "results" in result:
            for r in result["results"]:
                if years and r["year"] not in years:
                    continue
                metrics.append(FinancialMetric(
                    name=metric_concept,
                    value=Decimal(str(r["value"])),
                    unit=r.get("unit", "USD"),
                    period=r.get("period", f"FY{r['year']}"),
                    source=Citation(
                        source_document=f"{r['ticker']} {r['year']} 10-K",
                        section="XBRL Data",
                        ticker=r["ticker"],
                        year=r["year"],
                        quote_snippet=f"{metric_concept}: {r['value']}",
                    ),
                ))
        companies[ticker.upper()] = metrics

    # Build analysis summary
    parts: list[str] = []
    for tk, metrics in companies.items():
        if metrics:
            vals = [f"{m.period}: {m.value} {m.unit}" for m in metrics]
            parts.append(f"{tk}: {', '.join(vals)}")
        else:
            parts.append(f"{tk}: no data")

    analysis = f"Comparison of {metric_concept} — " + "; ".join(parts)

    return CompanyComparison(
        metric_name=metric_concept,
        companies=companies,
        analysis=analysis,
    )


# ---------------------------------------------------------------------------
# Tool: Retrieve Context
# ---------------------------------------------------------------------------


def retrieve_context_tool(
    query: str,
    retriever: HybridRetriever,
    ticker: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve context from the hybrid retriever with formatted citations.

    Args:
        query: Natural language query.
        retriever: HybridRetriever instance.
        ticker: Optional ticker filter.
        top_k: Number of results to return.

    Returns:
        List of dicts with 'content', 'citation', and 'score' keys.
    """
    filters = {"ticker": ticker} if ticker else None
    results: list[SearchResult] = retriever.search(query, filters=filters)

    formatted: list[dict[str, Any]] = []
    for r in results[:top_k]:
        meta = r.metadata
        citation = Citation(
            source_document=f"{meta.ticker} {meta.year} {meta.filing_type}",
            section=meta.section_name,
            ticker=meta.ticker,
            year=meta.year,
            quote_snippet=r.content[:100],
        )
        formatted.append({
            "content": r.content,
            "citation": citation.model_dump(),
            "score": r.score,
            "chunk_id": r.chunk_id,
        })

    logger.info("Retrieved %d context chunks for query: %s", len(formatted), query[:80])
    return formatted


# ---------------------------------------------------------------------------
# Legacy wrappers (used by existing stubs)
# ---------------------------------------------------------------------------


def xbrl_lookup(
    ticker: str,
    concept: str,
    period: str | None = None,
) -> dict:
    """Look up an XBRL financial fact for a company.

    Args:
        ticker: Stock ticker symbol.
        concept: XBRL taxonomy concept (e.g., 'us-gaap:Revenue').
        period: Optional fiscal period filter (e.g., '2023-FY').

    Returns:
        Dictionary containing the XBRL fact value and metadata.
    """
    return xbrl_lookup_tool(ticker, concept, period)


def calculate_ratio(
    numerator: float,
    denominator: float,
    metric_name: str = "ratio",
) -> FinancialCalculation:
    """Calculate a financial ratio from two values.

    Args:
        numerator: The numerator value.
        denominator: The denominator value.
        metric_name: Name for the resulting metric.

    Returns:
        FinancialCalculation with the computed ratio.
    """
    if denominator == 0:
        return FinancialCalculation(
            metric_name=metric_name,
            value=0.0,
            inputs={"numerator": numerator, "denominator": denominator},
            formula=f"{metric_name} = {numerator} / {denominator} (division by zero)",
        )
    value = numerator / denominator
    return FinancialCalculation(
        metric_name=metric_name,
        value=round(value, 4),
        inputs={"numerator": numerator, "denominator": denominator},
        formula=f"{metric_name} = {numerator} / {denominator}",
    )


def calculate_growth_rate(
    current_value: float,
    prior_value: float,
    metric_name: str = "growth_rate",
) -> FinancialCalculation:
    """Calculate year-over-year growth rate.

    Args:
        current_value: Current period value.
        prior_value: Prior period value.
        metric_name: Name for the resulting metric.

    Returns:
        FinancialCalculation with the computed growth rate as a percentage.
    """
    if prior_value == 0:
        return FinancialCalculation(
            metric_name=metric_name,
            value=0.0,
            unit="%",
            inputs={"current_value": current_value, "prior_value": prior_value},
            formula=f"{metric_name} = (current - prior) / prior * 100 (division by zero)",
        )
    rate = ((current_value - prior_value) / prior_value) * 100
    return FinancialCalculation(
        metric_name=metric_name,
        value=round(rate, 4),
        unit="%",
        inputs={"current_value": current_value, "prior_value": prior_value},
        formula=f"{metric_name} = ({current_value} - {prior_value}) / {prior_value} * 100",
    )


def calculate_margins(
    revenue: float,
    cost_of_revenue: float,
    operating_expenses: float,
    net_income: float,
) -> list[FinancialCalculation]:
    """Calculate gross, operating, and net profit margins.

    Args:
        revenue: Total revenue.
        cost_of_revenue: Cost of goods sold / cost of revenue.
        operating_expenses: Total operating expenses.
        net_income: Net income.

    Returns:
        List of FinancialCalculation objects for each margin type.
    """
    if revenue == 0:
        return [
            FinancialCalculation(metric_name=name, value=0.0, unit="%",
                                 formula=f"{name}: division by zero (revenue=0)")
            for name in ("gross_margin", "operating_margin", "net_margin")
        ]

    gross_profit = revenue - cost_of_revenue
    operating_income = gross_profit - operating_expenses

    return [
        FinancialCalculation(
            metric_name="gross_margin",
            value=round((gross_profit / revenue) * 100, 4),
            unit="%",
            inputs={"revenue": revenue, "cost_of_revenue": cost_of_revenue},
            formula="gross_margin = (revenue - COGS) / revenue * 100",
        ),
        FinancialCalculation(
            metric_name="operating_margin",
            value=round((operating_income / revenue) * 100, 4),
            unit="%",
            inputs={"revenue": revenue, "operating_expenses": operating_expenses,
                    "gross_profit": gross_profit},
            formula="operating_margin = (gross_profit - opex) / revenue * 100",
        ),
        FinancialCalculation(
            metric_name="net_margin",
            value=round((net_income / revenue) * 100, 4),
            unit="%",
            inputs={"revenue": revenue, "net_income": net_income},
            formula="net_margin = net_income / revenue * 100",
        ),
    ]
