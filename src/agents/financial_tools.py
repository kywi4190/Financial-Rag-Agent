"""Agent tools for XBRL lookup and financial calculations.

Provides tool functions that can be bound to a LlamaIndex agent
for structured financial data retrieval and ratio calculations.
"""

from src.agents.models import FinancialCalculation


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
    ...


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
    ...


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
    ...


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
    ...
