"""Tests for the agents package (query engine, tools, models, memo generator)."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.financial_tools import (
    calculate_growth_rate,
    calculate_margins,
    calculate_ratio,
    calculate_ratio_tool,
    compare_metrics_tool,
    get_xbrl_store,
    register_xbrl_data,
    retrieve_context_tool,
    xbrl_lookup_tool,
)
from src.agents.memo_generator import MemoGenerator, _deduplicate_citations, _format_metric_value
from src.agents.models import (
    AnswerWithCitations,
    Citation,
    CompanyComparison,
    FinancialCalculation,
    FinancialMetric,
    InvestmentMemo,
    MemoSection,
    QueryResponse,
    SourceAttribution,
)
from src.chunking.models import ChunkMetadata
from src.retrieval.models import SearchResult


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_xbrl_store() -> None:
    """Clear the XBRL store before each test."""
    store = get_xbrl_store()
    store.clear()


@pytest.fixture()
def sample_xbrl_df() -> pd.DataFrame:
    """XBRL DataFrame for AAPL FY2024."""
    return pd.DataFrame([
        {"concept": "us-gaap:Revenues", "value": 383285000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:NetIncomeLoss", "value": 93736000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:Assets", "value": 352583000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:Liabilities", "value": 290437000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:StockholdersEquity", "value": 62146000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:GrossProfit", "value": 172933000000, "unit": "USD", "period": "FY2024"},
    ])


@pytest.fixture()
def sample_xbrl_df_msft() -> pd.DataFrame:
    """XBRL DataFrame for MSFT FY2024."""
    return pd.DataFrame([
        {"concept": "us-gaap:Revenues", "value": 211915000000, "unit": "USD", "period": "FY2024"},
        {"concept": "us-gaap:NetIncomeLoss", "value": 72361000000, "unit": "USD", "period": "FY2024"},
    ])


@pytest.fixture()
def mock_retriever() -> MagicMock:
    """Mock HybridRetriever that returns sample results."""
    retriever = MagicMock()
    retriever.search.return_value = [
        SearchResult(
            chunk_id="aapl-2024-mda-0",
            content="Total net revenue was $383.3 billion for fiscal year 2024.",
            score=0.85,
            metadata=ChunkMetadata(
                ticker="AAPL", year=2024, filing_type="10-K",
                section_name="Item 7. MD&A", chunk_index=0,
            ),
            source="hybrid",
        ),
        SearchResult(
            chunk_id="aapl-2024-risk-0",
            content="Global market conditions could materially adversely affect the Company.",
            score=0.72,
            metadata=ChunkMetadata(
                ticker="AAPL", year=2024, filing_type="10-K",
                section_name="Item 1A. Risk Factors", chunk_index=0,
            ),
            source="hybrid",
        ),
    ]
    return retriever


# ── Model Tests ─────────────────────────────────────────────────────────────


class TestCitation:
    """Tests for the Citation model."""

    def test_citation_creation(self) -> None:
        c = Citation(
            source_document="AAPL 2024 10-K",
            section="Item 7. MD&A",
            ticker="AAPL",
            year=2024,
            quote_snippet="Revenue was $383.3 billion",
        )
        assert c.ticker == "AAPL"
        assert c.year == 2024

    def test_citation_snippet_max_length(self) -> None:
        with pytest.raises(Exception):
            Citation(
                source_document="AAPL 2024 10-K",
                section="Item 7",
                ticker="AAPL",
                year=2024,
                quote_snippet="x" * 101,
            )


class TestAnswerWithCitations:
    """Tests for AnswerWithCitations model."""

    def test_confidence_bounds(self) -> None:
        a = AnswerWithCitations(answer="test", confidence=0.5)
        assert a.confidence == 0.5

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(Exception):
            AnswerWithCitations(answer="test", confidence=1.5)


class TestFinancialMetric:
    """Tests for FinancialMetric model."""

    def test_decimal_value(self) -> None:
        m = FinancialMetric(
            name="Revenue",
            value=Decimal("383285000000"),
            unit="USD",
            period="FY2024",
            source=Citation(
                source_document="AAPL 2024 10-K", section="XBRL",
                ticker="AAPL", year=2024, quote_snippet="Revenue",
            ),
        )
        assert m.value == Decimal("383285000000")


class TestInvestmentMemo:
    """Tests for InvestmentMemo model."""

    def test_memo_creation(self) -> None:
        section = MemoSection(title="Test", content="Content", citations=[])
        memo = InvestmentMemo(
            ticker="AAPL",
            company_name="Apple Inc.",
            executive_summary=section,
            company_overview=section,
            financial_highlights=section,
            risk_factors=section,
            mda_synthesis=section,
            bull_bear_case=section,
        )
        assert memo.ticker == "AAPL"
        assert isinstance(memo.date_generated, datetime)


# ── Financial Tools Tests ───────────────────────────────────────────────────


class TestCalculateRatio:
    """Tests for calculate_ratio."""

    def test_basic_ratio(self) -> None:
        result = calculate_ratio(100.0, 50.0, "test_ratio")
        assert result.value == 2.0
        assert result.metric_name == "test_ratio"

    def test_division_by_zero(self) -> None:
        result = calculate_ratio(100.0, 0.0, "bad_ratio")
        assert result.value == 0.0
        assert "division by zero" in result.formula


class TestCalculateGrowthRate:
    """Tests for calculate_growth_rate."""

    def test_positive_growth(self) -> None:
        result = calculate_growth_rate(120.0, 100.0)
        assert result.value == 20.0
        assert result.unit == "%"

    def test_negative_growth(self) -> None:
        result = calculate_growth_rate(80.0, 100.0)
        assert result.value == -20.0

    def test_zero_prior(self) -> None:
        result = calculate_growth_rate(100.0, 0.0)
        assert result.value == 0.0


class TestCalculateMargins:
    """Tests for calculate_margins."""

    def test_margin_calculations(self) -> None:
        results = calculate_margins(
            revenue=1000.0,
            cost_of_revenue=600.0,
            operating_expenses=100.0,
            net_income=200.0,
        )
        assert len(results) == 3

        gross = results[0]
        assert gross.metric_name == "gross_margin"
        assert gross.value == 40.0

        operating = results[1]
        assert operating.metric_name == "operating_margin"
        assert operating.value == 30.0

        net = results[2]
        assert net.metric_name == "net_margin"
        assert net.value == 20.0

    def test_zero_revenue(self) -> None:
        results = calculate_margins(0.0, 0.0, 0.0, 0.0)
        assert all(r.value == 0.0 for r in results)


class TestXbrlLookupTool:
    """Tests for xbrl_lookup_tool."""

    def test_lookup_found(self, sample_xbrl_df: pd.DataFrame) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        result = xbrl_lookup_tool("AAPL", "Revenues")
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["value"] == 383285000000

    def test_lookup_not_found_ticker(self) -> None:
        result = xbrl_lookup_tool("ZZZZ", "Revenues")
        assert "error" in result

    def test_lookup_not_found_concept(self, sample_xbrl_df: pd.DataFrame) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        result = xbrl_lookup_tool("AAPL", "NonexistentConcept")
        assert "error" in result

    def test_lookup_with_period_filter(self, sample_xbrl_df: pd.DataFrame) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        result = xbrl_lookup_tool("AAPL", "Revenues", period="FY2024")
        assert "results" in result
        assert len(result["results"]) == 1

    def test_case_insensitive_ticker(self, sample_xbrl_df: pd.DataFrame) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        result = xbrl_lookup_tool("aapl", "Revenues")
        assert "results" in result


class TestCalculateRatioTool:
    """Tests for calculate_ratio_tool."""

    def test_with_explicit_values(self) -> None:
        result = calculate_ratio_tool("AAPL", "debt_to_equity", numerator=200.0, denominator=100.0)
        assert result.value == 2.0

    def test_from_xbrl(self, sample_xbrl_df: pd.DataFrame) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        result = calculate_ratio_tool("AAPL", "gross_margin", year=2024)
        assert result.value > 0
        assert result.unit == "%"

    def test_unknown_ratio(self) -> None:
        result = calculate_ratio_tool("AAPL", "unknown_ratio_xyz")
        assert result.value == 0.0
        assert "Unknown ratio" in result.formula


class TestCompareMetricsTool:
    """Tests for compare_metrics_tool."""

    def test_compare_two_companies(
        self,
        sample_xbrl_df: pd.DataFrame,
        sample_xbrl_df_msft: pd.DataFrame,
    ) -> None:
        register_xbrl_data("AAPL", 2024, sample_xbrl_df)
        register_xbrl_data("MSFT", 2024, sample_xbrl_df_msft)
        result = compare_metrics_tool("Revenues", ["AAPL", "MSFT"])
        assert isinstance(result, CompanyComparison)
        assert "AAPL" in result.companies
        assert "MSFT" in result.companies
        assert len(result.companies["AAPL"]) >= 1

    def test_compare_no_data(self) -> None:
        result = compare_metrics_tool("Revenues", ["ZZZZ"])
        assert result.companies["ZZZZ"] == []


class TestRetrieveContextTool:
    """Tests for retrieve_context_tool."""

    def test_returns_formatted_context(self, mock_retriever: MagicMock) -> None:
        results = retrieve_context_tool("What is AAPL revenue?", mock_retriever)
        assert len(results) == 2
        assert "content" in results[0]
        assert "citation" in results[0]
        assert "score" in results[0]
        assert results[0]["citation"]["ticker"] == "AAPL"

    def test_passes_ticker_filter(self, mock_retriever: MagicMock) -> None:
        retrieve_context_tool("revenue", mock_retriever, ticker="AAPL")
        mock_retriever.search.assert_called_once_with(
            "revenue", filters={"ticker": "AAPL"}
        )

    def test_respects_top_k(self, mock_retriever: MagicMock) -> None:
        results = retrieve_context_tool("test", mock_retriever, top_k=1)
        assert len(results) == 1


# ── Query Engine Tests ──────────────────────────────────────────────────────


class TestFinancialQueryEngine:
    """Tests for FinancialQueryEngine."""

    @patch("src.agents.query_engine.get_settings")
    @patch("src.agents.query_engine.OpenAI")
    @patch("src.agents.query_engine.OpenAIEmbedding")
    def test_query_returns_answer_with_citations(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        mock_retriever: MagicMock,
    ) -> None:
        """Test that query returns an AnswerWithCitations."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_settings.return_value.embedding_model = "text-embedding-3-small"

        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text="0.9")
        mock_llm_cls.return_value = mock_llm

        from src.agents.query_engine import FinancialQueryEngine

        engine = FinancialQueryEngine(retriever=mock_retriever)
        # Override the LLM to control outputs
        engine._llm = mock_llm

        # First call: relevance score, Second call: answer
        mock_llm.complete.side_effect = [
            MagicMock(text="0.9"),  # relevance eval
            MagicMock(text="Apple revenue was $383.3 billion in FY2024."),  # answer
        ]

        result = engine.query("What was Apple's revenue in 2024?")
        assert isinstance(result, AnswerWithCitations)
        assert len(result.citations) > 0

    @patch("src.agents.query_engine.get_settings")
    @patch("src.agents.query_engine.OpenAI")
    @patch("src.agents.query_engine.OpenAIEmbedding")
    def test_crag_reformulates_on_low_confidence(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        mock_retriever: MagicMock,
    ) -> None:
        """Test CRAG reformulates query when confidence < 0.6."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_settings.return_value.embedding_model = "text-embedding-3-small"

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        from src.agents.query_engine import FinancialQueryEngine

        engine = FinancialQueryEngine(retriever=mock_retriever)
        engine._llm = mock_llm

        mock_llm.complete.side_effect = [
            MagicMock(text="0.3"),  # low relevance -> triggers reformulation
            MagicMock(text="What was Apple Inc total revenue for fiscal year 2024?"),  # reformulation
            MagicMock(text="0.8"),  # second relevance eval
            MagicMock(text="Revenue was $383.3 billion."),  # final answer
        ]

        result = engine.query("AAPL rev 24?")
        assert isinstance(result, AnswerWithCitations)
        # Retriever should be called twice (initial + re-retrieve)
        assert mock_retriever.search.call_count == 2

    @patch("src.agents.query_engine.get_settings")
    @patch("src.agents.query_engine.OpenAI")
    @patch("src.agents.query_engine.OpenAIEmbedding")
    def test_query_with_filters_returns_query_response(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        mock_retriever: MagicMock,
    ) -> None:
        """Test query_with_filters returns a QueryResponse."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_settings.return_value.embedding_model = "text-embedding-3-small"

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        from src.agents.query_engine import FinancialQueryEngine

        engine = FinancialQueryEngine(retriever=mock_retriever)
        engine._llm = mock_llm

        mock_llm.complete.side_effect = [
            MagicMock(text="0.85"),
            MagicMock(text="Apple revenue was $383 billion."),
        ]

        result = engine.query_with_filters(
            "What was revenue?", ticker="AAPL", filing_type="10-K"
        )
        assert isinstance(result, QueryResponse)
        assert result.query == "What was revenue?"
        assert result.confidence is not None

    @patch("src.agents.query_engine.get_settings")
    @patch("src.agents.query_engine.OpenAI")
    @patch("src.agents.query_engine.OpenAIEmbedding")
    def test_empty_retrieval_returns_low_confidence(
        self,
        mock_embed_cls: MagicMock,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test that no retriever yields a low-confidence answer."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_settings.return_value.embedding_model = "text-embedding-3-small"

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        from src.agents.query_engine import FinancialQueryEngine

        engine = FinancialQueryEngine(retriever=None)
        engine._llm = mock_llm

        # No retriever -> no context -> confidence 0.0 returned without LLM call
        result = engine.query("Anything?")
        assert isinstance(result, AnswerWithCitations)
        assert result.confidence == 0.0


# ── Memo Generator Tests ──────────────────────────────────────────────────


class TestFormatMetricValue:
    """Tests for _format_metric_value helper."""

    def test_large_number(self) -> None:
        assert _format_metric_value(383285000000) == "383,285,000,000"

    def test_small_number(self) -> None:
        assert _format_metric_value(6.52) == "6.52"

    def test_string_fallback(self) -> None:
        assert _format_metric_value("N/A") == "N/A"


class TestDeduplicateCitations:
    """Tests for _deduplicate_citations helper."""

    def test_removes_duplicates(self) -> None:
        c1 = Citation(
            source_document="AAPL 2024 10-K", section="XBRL Data",
            ticker="AAPL", year=2024, quote_snippet="Revenue",
        )
        c2 = Citation(
            source_document="AAPL 2024 10-K", section="XBRL Data",
            ticker="AAPL", year=2024, quote_snippet="Net Income",
        )
        c3 = Citation(
            source_document="AAPL 2024 10-K", section="Item 7. MD&A",
            ticker="AAPL", year=2024, quote_snippet="Forward outlook",
        )
        result = _deduplicate_citations([c1, c2, c3])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert _deduplicate_citations([]) == []


class TestToMarkdown:
    """Tests for InvestmentMemo.to_markdown()."""

    def test_contains_all_sections(self) -> None:
        section = MemoSection(title="Test", content="Content", citations=[])
        memo = InvestmentMemo(
            ticker="AAPL",
            company_name="Apple Inc.",
            executive_summary=section,
            company_overview=section,
            financial_highlights=section,
            risk_factors=section,
            mda_synthesis=section,
            bull_bear_case=section,
        )
        md = memo.to_markdown()
        assert "# Investment Memo: AAPL — Apple Inc." in md
        assert "## Test" in md
        assert "Content" in md

    def test_includes_citations_when_present(self) -> None:
        citation = Citation(
            source_document="AAPL 2024 10-K", section="XBRL Data",
            ticker="AAPL", year=2024, quote_snippet="Revenue: 383B",
        )
        section_with = MemoSection(title="Highlights", content="Data", citations=[citation])
        section_without = MemoSection(title="Other", content="Text", citations=[])
        memo = InvestmentMemo(
            ticker="AAPL",
            company_name="Apple Inc.",
            executive_summary=section_with,
            company_overview=section_without,
            financial_highlights=section_without,
            risk_factors=section_without,
            mda_synthesis=section_without,
            bull_bear_case=section_without,
        )
        md = memo.to_markdown()
        assert "### Sources" in md
        assert "AAPL 2024 10-K" in md


class TestMemoGenerator:
    """Tests for the MemoGenerator class."""

    @pytest.fixture()
    def apple_xbrl_data(self) -> pd.DataFrame:
        """Register comprehensive Apple XBRL data and return the DataFrame."""
        df = pd.DataFrame([
            {"concept": "us-gaap:Revenues", "value": 383285000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:NetIncomeLoss", "value": 93736000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:GrossProfit", "value": 172933000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:Assets", "value": 352583000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:Liabilities", "value": 290437000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:StockholdersEquity", "value": 62146000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:AssetsCurrent", "value": 152987000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:LiabilitiesCurrent", "value": 176392000000, "unit": "USD", "period": "FY2024"},
            {"concept": "us-gaap:EarningsPerShareBasic", "value": 6.08, "unit": "USD/shares", "period": "FY2024"},
            {"concept": "us-gaap:CommonStockSharesOutstanding", "value": 15408095000, "unit": "shares", "period": "FY2024"},
        ])
        register_xbrl_data("AAPL", 2024, df)
        return df

    @pytest.fixture()
    def memo_retriever(self) -> MagicMock:
        """Mock retriever returning results for different section queries."""
        retriever = MagicMock()

        def _mock_search(query: str, filters: dict | None = None) -> list[SearchResult]:
            if "risk" in query.lower():
                return [SearchResult(
                    chunk_id="aapl-2024-risk-0",
                    content="The Company faces significant competition and rapid technological change.",
                    score=0.82,
                    metadata=ChunkMetadata(
                        ticker="AAPL", year=2024, filing_type="10-K",
                        section_name="Item 1A. Risk Factors", chunk_index=0,
                    ),
                    source="hybrid",
                )]
            if "md&a" in query.lower() or "management" in query.lower():
                return [SearchResult(
                    chunk_id="aapl-2024-mda-0",
                    content="Revenue increased 2% year-over-year driven by Services growth.",
                    score=0.78,
                    metadata=ChunkMetadata(
                        ticker="AAPL", year=2024, filing_type="10-K",
                        section_name="Item 7. MD&A", chunk_index=0,
                    ),
                    source="hybrid",
                )]
            if "business" in query.lower() or "overview" in query.lower():
                return [SearchResult(
                    chunk_id="aapl-2024-bus-0",
                    content="Apple designs, manufactures, and markets smartphones, tablets, and computers.",
                    score=0.90,
                    metadata=ChunkMetadata(
                        ticker="AAPL", year=2024, filing_type="10-K",
                        section_name="Item 1. Business", chunk_index=0,
                    ),
                    source="hybrid",
                )]
            return []

        retriever.search.side_effect = _mock_search
        return retriever

    @patch("src.agents.memo_generator.get_settings")
    @patch("src.agents.memo_generator.OpenAI")
    def test_memo_structure_complete(
        self,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        apple_xbrl_data: pd.DataFrame,
        memo_retriever: MagicMock,
    ) -> None:
        """Verify all memo sections are present and well-formed."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        # 7 LLM calls: company name, financial highlights, overview, risks,
        # mda, executive summary, bull/bear
        mock_llm.complete.side_effect = [
            MagicMock(text="Apple Inc."),
            MagicMock(text="Apple reported $383.3B in revenue with 45.1% gross margin."),
            MagicMock(text="Apple designs and markets consumer electronics worldwide."),
            MagicMock(text="1. Competition (Market): Intense competition in all segments."),
            MagicMock(text="Management highlighted Services growth as a key driver."),
            MagicMock(text="Apple is a dominant consumer tech franchise with strong margins."),
            MagicMock(text="BULL CASE:\n1. Services growth\n2. Margin expansion\n3. Ecosystem\n\nBEAR CASE:\n1. China risk\n2. Regulatory\n3. Saturation"),
        ]

        generator = MemoGenerator(retriever=memo_retriever)
        memo = generator.generate_memo("AAPL", year=2024)

        assert isinstance(memo, InvestmentMemo)
        assert memo.ticker == "AAPL"
        assert memo.company_name == "Apple Inc."

        # All sections populated
        assert memo.executive_summary.content
        assert memo.company_overview.content
        assert memo.financial_highlights.content
        assert memo.risk_factors.content
        assert memo.mda_synthesis.content
        assert memo.bull_bear_case.content

        # Correct section titles
        assert memo.executive_summary.title == "Executive Summary"
        assert memo.financial_highlights.title == "Financial Highlights"
        assert memo.risk_factors.title == "Risk Factors"
        assert memo.mda_synthesis.title == "MD&A Synthesis"
        assert memo.company_overview.title == "Company Overview"
        assert memo.bull_bear_case.title == "Bull & Bear Case"

        # Produces valid markdown
        md = memo.to_markdown()
        assert "# Investment Memo: AAPL" in md

    @patch("src.agents.memo_generator.get_settings")
    @patch("src.agents.memo_generator.OpenAI")
    def test_all_sections_have_citations(
        self,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        apple_xbrl_data: pd.DataFrame,
        memo_retriever: MagicMock,
    ) -> None:
        """Verify every section has at least one citation."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_llm.complete.side_effect = [
            MagicMock(text="Apple Inc."),
            MagicMock(text="Financial highlights content."),
            MagicMock(text="Company overview content."),
            MagicMock(text="Risk factors content."),
            MagicMock(text="MD&A synthesis content."),
            MagicMock(text="Executive summary content."),
            MagicMock(text="Bull and bear case content."),
        ]

        generator = MemoGenerator(retriever=memo_retriever)
        memo = generator.generate_memo("AAPL", year=2024)

        sections = [
            memo.executive_summary,
            memo.company_overview,
            memo.financial_highlights,
            memo.risk_factors,
            memo.mda_synthesis,
            memo.bull_bear_case,
        ]

        for section in sections:
            assert len(section.citations) > 0, (
                f"Section '{section.title}' has no citations"
            )
            for citation in section.citations:
                assert citation.ticker == "AAPL"
                assert citation.source_document
                assert citation.section

    @patch("src.agents.memo_generator.get_settings")
    @patch("src.agents.memo_generator.OpenAI")
    def test_financial_metrics_accuracy(
        self,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
        apple_xbrl_data: pd.DataFrame,
    ) -> None:
        """Verify financial data agent extracts correct Apple metrics from XBRL."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        mock_llm.complete.return_value = MagicMock(text="Financial summary.")

        generator = MemoGenerator(retriever=None)
        section = generator._extract_financial_data("AAPL", year=2024)

        # Verify citations contain the known XBRL values
        citation_snippets = [c.quote_snippet for c in section.citations]
        snippet_text = " ".join(citation_snippets)

        # Revenue should be present
        assert any("Revenue" in s for s in citation_snippets)
        # Net Income should be present
        assert any("Net Income" in s for s in citation_snippets)
        # Gross Profit should be present
        assert any("Gross Profit" in s for s in citation_snippets)

        # Verify the LLM prompt contained the correct metric values
        llm_prompt = mock_llm.complete.call_args[0][0]
        assert "383,285,000,000" in llm_prompt  # Revenue
        assert "93,736,000,000" in llm_prompt   # Net Income
        assert "172,933,000,000" in llm_prompt  # Gross Profit

        # Verify ratios were calculated and included
        assert "gross_margin" in llm_prompt
        assert "net_margin" in llm_prompt

    @patch("src.agents.memo_generator.get_settings")
    @patch("src.agents.memo_generator.OpenAI")
    def test_handles_missing_xbrl_data(
        self,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Verify graceful handling when no XBRL data is available."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        generator = MemoGenerator(retriever=None)
        section = generator._extract_financial_data("ZZZZ", year=2024)

        assert section.title == "Financial Highlights"
        assert "not available" in section.content.lower()
        assert section.citations == []
        # LLM should NOT be called when there's no data
        mock_llm.complete.assert_not_called()

    @patch("src.agents.memo_generator.get_settings")
    @patch("src.agents.memo_generator.OpenAI")
    def test_handles_no_retriever(
        self,
        mock_llm_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Verify memo generation works without a retriever (XBRL-only)."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        mock_llm.complete.side_effect = [
            MagicMock(text="Unknown Corp"),
            MagicMock(text="No financial data."),  # won't be called if no XBRL
            MagicMock(text="No overview available."),  # won't be called
            MagicMock(text="No risks available."),
            MagicMock(text="No MD&A available."),
            MagicMock(text="Summary."),
            MagicMock(text="Bull/Bear."),
        ]

        generator = MemoGenerator(retriever=None)
        memo = generator.generate_memo("ZZZZ", year=2024)

        assert isinstance(memo, InvestmentMemo)
        assert memo.ticker == "ZZZZ"
