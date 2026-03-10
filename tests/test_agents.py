"""Tests for the agents package (query engine, tools, memo generator)."""

import pytest


class TestFinancialQueryEngine:
    """Tests for FinancialQueryEngine."""

    def test_query_returns_response(self) -> None:
        """Test that query returns a structured QueryResponse."""
        ...

    def test_query_with_filters_applies_ticker_filter(self) -> None:
        """Test that ticker filter restricts results to one company."""
        ...


class TestFinancialTools:
    """Tests for financial tool functions."""

    def test_calculate_ratio(self) -> None:
        """Test ratio calculation with known inputs."""
        ...

    def test_calculate_growth_rate(self) -> None:
        """Test growth rate calculation with known inputs."""
        ...

    def test_calculate_margins(self) -> None:
        """Test margin calculations produce correct percentages."""
        ...


class TestMemoGenerator:
    """Tests for MemoGenerator."""

    def test_generate_returns_investment_memo(self) -> None:
        """Test that generate produces a structured InvestmentMemo."""
        ...
