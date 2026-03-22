"""Multi-agent investment memo generator with citations.

Orchestrates three specialized agents — Financial Data, Qualitative Analysis,
and Synthesis — to produce structured investment memos from SEC filings.
"""

import logging
from typing import Any, Optional

from llama_index.llms.openai import OpenAI

from src.agents.financial_tools import (
    calculate_ratio_tool,
    retrieve_context_tool,
    xbrl_lookup_tool,
)
from src.agents.models import Citation, InvestmentMemo, MemoSection
from src.config import get_settings
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts for each agent
# ---------------------------------------------------------------------------

_FINANCIAL_HIGHLIGHTS_PROMPT = """\
You are a financial analyst. Given the following XBRL-extracted financial data \
for {ticker} ({period}), write a concise Financial Highlights section for an \
investment memo.

Data:
{metrics_text}

Ratios:
{ratios_text}

Instructions:
- Present key metrics in a clear, readable format
- Highlight notable trends or outliers
- Use precise figures — do not round excessively
- Keep to 2-3 paragraphs"""

_RISK_FACTORS_PROMPT = """\
You are a financial analyst reviewing SEC filing risk factors for {ticker}.

Retrieved risk factor excerpts:
{context}

Instructions:
- Identify the top 5 material risks
- Categorize each risk (e.g., Regulatory, Market, Operational, Financial, Legal)
- Summarize each risk in 1-2 sentences
- Cite the source section for each risk"""

_MDA_PROMPT = """\
You are a financial analyst reviewing MD&A sections from {ticker}'s SEC filings.

Retrieved MD&A excerpts:
{context}

Instructions:
- Summarize management's forward-looking commentary
- Highlight key business drivers and challenges mentioned
- Note any guidance or outlook statements
- Keep to 2-3 paragraphs"""

_COMPANY_OVERVIEW_PROMPT = """\
You are a financial analyst writing a company overview for {ticker}.

Retrieved business description excerpts:
{context}

Instructions:
- Describe the company's business model and primary revenue streams
- Note market position and competitive advantages
- Keep to 2-3 paragraphs"""

_EXECUTIVE_SUMMARY_PROMPT = """\
You are a senior investment analyst writing an executive summary for {ticker}.

Financial Highlights:
{financial_highlights}

Company Overview:
{company_overview}

Key Risks:
{risk_factors}

MD&A Insights:
{mda_synthesis}

Instructions:
- Write a 3-4 sentence investment thesis
- Include key financial metrics (revenue, margins, growth)
- State the overall investment posture (bullish, bearish, or neutral)
- Be specific and data-driven"""

_BULL_BEAR_PROMPT = """\
You are a senior investment analyst creating bull and bear cases for {ticker}.

Financial Highlights:
{financial_highlights}

Company Overview:
{company_overview}

Risk Factors:
{risk_factors}

MD&A Insights:
{mda_synthesis}

Instructions:
- Present exactly 3 bull case arguments, each with supporting evidence from filings
- Present exactly 3 bear case arguments, each with supporting evidence from filings
- Format as:
  BULL CASE:
  1. [Argument]: [Evidence]
  2. [Argument]: [Evidence]
  3. [Argument]: [Evidence]

  BEAR CASE:
  1. [Argument]: [Evidence]
  2. [Argument]: [Evidence]
  3. [Argument]: [Evidence]"""


def _format_metric_value(value: Any) -> str:
    """Format a metric value for display.

    Args:
        value: Raw metric value (int, float, or string).

    Returns:
        Formatted string representation.
    """
    try:
        num = float(value)
        if abs(num) >= 1000:
            return f"{num:,.0f}"
        return f"{num:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _deduplicate_citations(citations: list[Citation]) -> list[Citation]:
    """Remove duplicate citations by (source_document, section) pair.

    Args:
        citations: List of citations, possibly with duplicates.

    Returns:
        Deduplicated list preserving order.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Citation] = []
    for c in citations:
        key = (c.source_document, c.section)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


class MemoGenerator:
    """Multi-agent investment memo generator.

    Orchestrates three agents to produce a structured InvestmentMemo:
    1. Financial Data Agent — extracts and formats XBRL metrics
    2. Qualitative Analysis Agent — retrieves and synthesizes narrative sections
    3. Synthesis Agent — combines outputs into executive summary and bull/bear cases

    Args:
        retriever: HybridRetriever for document search.
        llm_model: OpenAI model identifier.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Initialize the memo generator with LLM and optional retriever."""
        settings = get_settings()
        self._llm_model = llm_model or settings.llm_model
        self._retriever = retriever
        self._llm = OpenAI(model=self._llm_model, temperature=0.1)

    def _llm_call(self, prompt: str) -> str:
        """Make an LLM call with input/output logging.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM response text.
        """
        logger.info("LLM call input: %s", prompt[:200])
        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
            logger.info("LLM call output: %s", text[:200])
            return text
        except Exception:
            logger.exception("LLM call failed")
            return ""

    def _retrieve_for_section(
        self,
        query: str,
        ticker: str,
        top_k: int = 5,
    ) -> tuple[str, list[Citation]]:
        """Retrieve context for a specific section query.

        Args:
            query: The retrieval query.
            ticker: Company ticker for filtering.
            top_k: Number of results.

        Returns:
            Tuple of (formatted context text, list of citations).
        """
        if self._retriever is None:
            return "", []

        chunks = retrieve_context_tool(
            query=query,
            retriever=self._retriever,
            ticker=ticker,
            top_k=top_k,
        )

        if not chunks:
            return "", []

        context_parts: list[str] = []
        citations: list[Citation] = []
        for c in chunks:
            source = c["citation"]
            context_parts.append(
                f"[{source['source_document']}, {source['section']}]\n{c['content']}"
            )
            citations.append(Citation(**source))

        return "\n\n---\n\n".join(context_parts), citations

    # ── Agent 1: Financial Data Agent ─────────────────────────────────────

    def _extract_financial_data(
        self,
        ticker: str,
        year: int | None = None,
    ) -> MemoSection:
        """Extract key financial metrics from XBRL data.

        Looks up revenue, net income, gross profit, assets, liabilities,
        equity, EPS, and shares outstanding. Calculates gross margin,
        net margin, debt-to-equity, and current ratio.

        Args:
            ticker: Stock ticker symbol.
            year: Optional fiscal year filter.

        Returns:
            MemoSection with financial highlights.
        """
        metrics: list[dict[str, Any]] = []
        citations: list[Citation] = []

        concepts = [
            ("Revenues", "Revenue"),
            ("NetIncomeLoss", "Net Income"),
            ("GrossProfit", "Gross Profit"),
            ("Assets", "Total Assets"),
            ("Liabilities", "Total Liabilities"),
            ("StockholdersEquity", "Stockholders' Equity"),
            ("AssetsCurrent", "Current Assets"),
            ("LiabilitiesCurrent", "Current Liabilities"),
            ("EarningsPerShareBasic", "EPS (Basic)"),
            ("CommonStockSharesOutstanding", "Shares Outstanding"),
        ]

        for concept, display_name in concepts:
            result = xbrl_lookup_tool(ticker, concept)
            if "results" not in result:
                continue
            for r in result["results"]:
                if year and r["year"] != year:
                    continue
                metrics.append({
                    "name": display_name,
                    "value": r["value"],
                    "unit": r.get("unit", "USD"),
                    "period": r.get("period", f"FY{r['year']}"),
                })
                citations.append(Citation(
                    source_document=f"{r['ticker']} 10-K {r['year']}",
                    section="XBRL Data",
                    ticker=r["ticker"],
                    year=r["year"],
                    quote_snippet=f"{display_name}: {_format_metric_value(r['value'])}"[:100],
                ))
                break  # Take first match per concept

        # Calculate ratios
        ratio_names = ["gross_margin", "net_margin", "debt_to_equity", "current_ratio"]
        ratios = []
        for ratio_name in ratio_names:
            calc = calculate_ratio_tool(ticker, ratio_name, year=year)
            if calc.value != 0.0:
                ratios.append(calc)

        # Format for LLM
        metrics_text = "\n".join(
            f"- {m['name']}: {_format_metric_value(m['value'])} {m['unit']} ({m['period']})"
            for m in metrics
        ) or "No XBRL data available."

        ratios_text = "\n".join(
            f"- {r.metric_name}: {r.value:.2f}{r.unit}"
            for r in ratios
        ) or "No ratio data available."

        if metrics or ratios:
            content = self._llm_call(
                _FINANCIAL_HIGHLIGHTS_PROMPT.format(
                    ticker=ticker,
                    period=f"FY{year}" if year else "latest available",
                    metrics_text=metrics_text,
                    ratios_text=ratios_text,
                )
            )
        else:
            content = "Financial data not available for this company."

        return MemoSection(
            title="Financial Highlights",
            content=content or "Financial data could not be summarized.",
            citations=citations,
        )

    # ── Agent 2: Qualitative Analysis Agent ───────────────────────────────

    def _extract_risk_factors(self, ticker: str) -> MemoSection:
        """Extract and synthesize risk factors from Item 1A.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            MemoSection with categorized risk factors.
        """
        context, citations = self._retrieve_for_section(
            f"{ticker} risk factors material risks Item 1A",
            ticker=ticker,
        )

        if context:
            content = self._llm_call(
                _RISK_FACTORS_PROMPT.format(ticker=ticker, context=context)
            )
        else:
            content = "Risk factor data not available from retrieved filings."

        return MemoSection(
            title="Risk Factors",
            content=content or "Risk factors could not be synthesized.",
            citations=citations,
        )

    def _extract_mda(self, ticker: str) -> MemoSection:
        """Extract and synthesize MD&A commentary from Item 7.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            MemoSection with MD&A synthesis.
        """
        context, citations = self._retrieve_for_section(
            f"{ticker} management discussion analysis outlook forward looking Item 7 MD&A",
            ticker=ticker,
        )

        if context:
            content = self._llm_call(
                _MDA_PROMPT.format(ticker=ticker, context=context)
            )
        else:
            content = "MD&A data not available from retrieved filings."

        return MemoSection(
            title="MD&A Synthesis",
            content=content or "MD&A could not be synthesized.",
            citations=citations,
        )

    def _extract_company_overview(self, ticker: str) -> MemoSection:
        """Extract business description from Item 1.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            MemoSection with company overview.
        """
        context, citations = self._retrieve_for_section(
            f"{ticker} business description company overview products services Item 1",
            ticker=ticker,
        )

        if context:
            content = self._llm_call(
                _COMPANY_OVERVIEW_PROMPT.format(ticker=ticker, context=context)
            )
        else:
            content = "Company overview not available from retrieved filings."

        return MemoSection(
            title="Company Overview",
            content=content or "Company overview could not be generated.",
            citations=citations,
        )

    # ── Agent 3: Synthesis Agent ──────────────────────────────────────────

    def _synthesize_executive_summary(
        self,
        ticker: str,
        financial_highlights: MemoSection,
        company_overview: MemoSection,
        risk_factors: MemoSection,
        mda_synthesis: MemoSection,
    ) -> MemoSection:
        """Generate executive summary from all sections.

        Args:
            ticker: Stock ticker symbol.
            financial_highlights: Financial highlights section.
            company_overview: Company overview section.
            risk_factors: Risk factors section.
            mda_synthesis: MD&A section.

        Returns:
            MemoSection with executive summary.
        """
        content = self._llm_call(
            _EXECUTIVE_SUMMARY_PROMPT.format(
                ticker=ticker,
                financial_highlights=financial_highlights.content,
                company_overview=company_overview.content,
                risk_factors=risk_factors.content,
                mda_synthesis=mda_synthesis.content,
            )
        )

        all_citations: list[Citation] = []
        for section in [financial_highlights, company_overview, risk_factors, mda_synthesis]:
            all_citations.extend(section.citations)

        return MemoSection(
            title="Executive Summary",
            content=content or "Executive summary could not be generated.",
            citations=_deduplicate_citations(all_citations),
        )

    def _synthesize_bull_bear(
        self,
        ticker: str,
        financial_highlights: MemoSection,
        company_overview: MemoSection,
        risk_factors: MemoSection,
        mda_synthesis: MemoSection,
    ) -> MemoSection:
        """Generate bull and bear cases from all sections.

        Args:
            ticker: Stock ticker symbol.
            financial_highlights: Financial highlights section.
            company_overview: Company overview section.
            risk_factors: Risk factors section.
            mda_synthesis: MD&A section.

        Returns:
            MemoSection with bull and bear cases.
        """
        content = self._llm_call(
            _BULL_BEAR_PROMPT.format(
                ticker=ticker,
                financial_highlights=financial_highlights.content,
                company_overview=company_overview.content,
                risk_factors=risk_factors.content,
                mda_synthesis=mda_synthesis.content,
            )
        )

        all_citations: list[Citation] = []
        for section in [financial_highlights, company_overview, risk_factors, mda_synthesis]:
            all_citations.extend(section.citations)

        return MemoSection(
            title="Bull & Bear Case",
            content=content or "Bull and bear cases could not be generated.",
            citations=_deduplicate_citations(all_citations),
        )

    # ── Main orchestration ────────────────────────────────────────────────

    def _resolve_company_name(self, ticker: str) -> str:
        """Resolve ticker to company name via LLM.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Company legal name.
        """
        name = self._llm_call(
            f"What is the legal company name for stock ticker {ticker}? "
            f"Respond with ONLY the company name, nothing else."
        )
        return name or ticker

    def generate_memo(
        self,
        ticker: str,
        year: Optional[int] = None,
    ) -> InvestmentMemo:
        """Generate a complete investment memo for a company.

        Orchestrates three agents:
        1. Financial Data Agent — XBRL metrics extraction
        2. Qualitative Analysis Agent — retrieval-based narrative analysis
        3. Synthesis Agent — executive summary and bull/bear cases

        Args:
            ticker: Stock ticker symbol.
            year: Optional fiscal year to focus on.

        Returns:
            Fully populated InvestmentMemo.
        """
        ticker = ticker.upper()
        logger.info("Generating investment memo for %s (year=%s)", ticker, year)

        # Resolve company name
        company_name = self._resolve_company_name(ticker)

        # Agent 1: Financial Data
        logger.info("Running Financial Data Agent for %s", ticker)
        financial_highlights = self._extract_financial_data(ticker, year)

        # Agent 2: Qualitative Analysis (3 sub-queries)
        logger.info("Running Qualitative Analysis Agent for %s", ticker)
        company_overview = self._extract_company_overview(ticker)
        risk_factors = self._extract_risk_factors(ticker)
        mda_synthesis = self._extract_mda(ticker)

        # Agent 3: Synthesis
        logger.info("Running Synthesis Agent for %s", ticker)
        executive_summary = self._synthesize_executive_summary(
            ticker, financial_highlights, company_overview,
            risk_factors, mda_synthesis,
        )
        bull_bear_case = self._synthesize_bull_bear(
            ticker, financial_highlights, company_overview,
            risk_factors, mda_synthesis,
        )

        memo = InvestmentMemo(
            ticker=ticker,
            company_name=company_name,
            executive_summary=executive_summary,
            company_overview=company_overview,
            financial_highlights=financial_highlights,
            risk_factors=risk_factors,
            mda_synthesis=mda_synthesis,
            bull_bear_case=bull_bear_case,
        )

        logger.info("Investment memo generated for %s", ticker)
        return memo
