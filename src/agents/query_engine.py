"""LlamaIndex query engine with Corrective RAG for financial document Q&A.

Wraps LlamaIndex SubQuestionQueryEngine with the project's hybrid
retrieval pipeline. Routes numerical queries to XBRL tools and
narrative queries to retrieval. Implements CRAG: if initial retrieval
confidence is below threshold, reformulates and re-retrieves.
"""

import logging
import re
from typing import Any

from llama_index.core import PromptTemplate
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.agents.financial_tools import (
    calculate_ratio_tool,
    compare_metrics_tool,
    retrieve_context_tool,
    xbrl_lookup_tool,
)
from src.agents.models import AnswerWithCitations, Citation, QueryResponse, SourceAttribution
from src.config import get_settings
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

FINANCIAL_SYSTEM_PROMPT = """\
You are a financial analyst assistant that answers questions about SEC filings.

Rules:
- Be precise with financial data — never fabricate or approximate numbers.
- Always cite sources: include the document name, section, ticker, and year.
- If the retrieved context does not contain enough information to answer
  confidently, say so explicitly rather than guessing.
- When comparing metrics across companies or periods, present data in a
  structured format.
- For numerical questions, prefer XBRL data when available over narrative text.
"""

CRAG_CONFIDENCE_THRESHOLD = 0.5

REFORMULATION_TEMPLATE = PromptTemplate(
    "The following query did not retrieve sufficiently relevant context:\n"
    "Original query: {original_query}\n\n"
    "Please reformulate this query to be more specific and likely to match "
    "relevant SEC filing content. Return only the reformulated query."
)


class FinancialQueryEngine:
    """Query engine for answering questions about financial filings.

    Integrates the hybrid retrieval pipeline with LlamaIndex's
    SubQuestionQueryEngine for multi-step decomposition. Implements
    Corrective RAG (CRAG) for low-confidence retrievals.

    Args:
        retriever: HybridRetriever instance for context retrieval.
        llm_model: OpenAI model identifier for generation.
        embedding_model: OpenAI model identifier for embeddings.
    """

    _COMPANY_TICKERS: dict[str, str] = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "alphabet": "GOOGL",
        "google": "GOOGL",
    }

    _KNOWN_TICKERS: set[str] = {"AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"}

    @staticmethod
    def _detect_query_type(query: str) -> str:
        """Classify a query as numerical, comparative, or analytical."""
        query_lower = query.lower()

        # Comparative signals
        comparative_patterns = [
            r"\bcompare\b", r"\bcompar", r"\bvs\.?\b", r"\bversus\b",
            r"\bdifference between\b", r"\bhow did .+ change\b",
            r"\bhow did .+ compare\b", r"\bfrom .+ to\b",
            r"\bgrew|growth|increased|decreased\b",
        ]
        comparative_count = sum(
            1 for p in comparative_patterns if re.search(p, query_lower)
        )
        if comparative_count >= 1:
            return "comparative"

        # Numerical signals
        numerical_patterns = [
            r"\bhow much\b", r"\bhow many\b", r"\bwhat was\b", r"\bwhat is\b",
            r"\bwhat were\b", r"\$", r"\d+", r"\beps\b", r"\brevenue\b",
            r"\bincome\b", r"\bmargin\b", r"\bratio\b",
        ]
        numerical_count = sum(
            1 for p in numerical_patterns if re.search(p, query_lower)
        )
        if numerical_count >= 2:
            return "numerical"

        return "analytical"

    @staticmethod
    def _extract_ticker(query: str) -> str | None:
        """Extract a single ticker from a natural language query.

        Returns None if the query mentions multiple companies (comparative)
        or no recognizable company.
        """
        query_lower = query.lower()

        # Check for direct ticker symbols (uppercase, 1-5 chars)
        direct_tickers = re.findall(r'\b([A-Z]{1,5})\b', query)
        found_tickers = [t for t in direct_tickers if t in FinancialQueryEngine._KNOWN_TICKERS]

        # Check for company names
        found_companies = [
            ticker
            for name, ticker in FinancialQueryEngine._COMPANY_TICKERS.items()
            if name in query_lower
        ]

        all_found = list(set(found_tickers + found_companies))

        # Only filter if exactly one company is mentioned
        if len(all_found) == 1:
            return all_found[0]
        return None

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the query engine with LLM and embedding models."""
        settings = get_settings()
        self._llm_model = llm_model or settings.llm_model
        self._embedding_model = embedding_model or settings.embedding_model
        self._retriever = retriever

        self._llm = OpenAI(model=self._llm_model, temperature=0.1)
        self._embed = OpenAIEmbedding(model_name=self._embedding_model)

        self._tools = self._build_tools()

    def _build_tools(self) -> list[FunctionTool]:
        """Build LlamaIndex FunctionTools for the agent."""
        tools: list[FunctionTool] = [
            FunctionTool.from_defaults(
                fn=xbrl_lookup_tool,
                name="xbrl_lookup",
                description=(
                    "Look up specific XBRL financial metrics (revenue, net income, "
                    "assets, liabilities, etc.) for a company by ticker and concept name."
                ),
            ),
            FunctionTool.from_defaults(
                fn=calculate_ratio_tool,
                name="calculate_ratio",
                description=(
                    "Calculate financial ratios (debt_to_equity, current_ratio, "
                    "gross_margin, net_margin, return_on_assets, return_on_equity) "
                    "from XBRL data or provided values."
                ),
            ),
            FunctionTool.from_defaults(
                fn=compare_metrics_tool,
                name="compare_metrics",
                description=(
                    "Compare a financial metric across multiple companies and/or years. "
                    "Provide a metric concept and list of tickers."
                ),
            ),
        ]
        return tools

    def _retrieve_context(
        self,
        query: str,
        ticker: str | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve context using the hybrid retriever."""
        if self._retriever is None:
            return []
        return retrieve_context_tool(
            query=query,
            retriever=self._retriever,
            ticker=ticker,
            top_k=top_k,
        )

    def _evaluate_context_relevance(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> float:
        """Evaluate how relevant the retrieved context is to the query.

        Uses the LLM to score relevance on a 0-1 scale.

        Args:
            query: The user query.
            context_chunks: Retrieved context dicts.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not context_chunks:
            return 0.0

        context_text = "\n\n".join(
            c["content"][:2000] for c in context_chunks[:5]
        )

        prompt = (
            f"On a scale of 0.0 to 1.0, how relevant is the following context "
            f"for answering this query?\n\n"
            f"Query: {query}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Respond with ONLY a decimal number between 0.0 and 1.0."
        )

        try:
            response = self._llm.complete(prompt)
            score = float(response.text.strip())
            logger.info("Context relevance score for '%s': %.2f", query[:50], score)
            return max(0.0, min(1.0, score))
        except (ValueError, Exception):
            logger.warning("Failed to evaluate context relevance, defaulting to 0.5")
            return 0.5

    def _reformulate_query(self, original_query: str) -> str:
        """Use the LLM to reformulate a query for better retrieval.

        Args:
            original_query: The query that produced low-confidence results.

        Returns:
            Reformulated query string.
        """
        prompt = REFORMULATION_TEMPLATE.format(original_query=original_query)
        try:
            response = self._llm.complete(prompt)
            reformulated = response.text.strip()
            logger.info(
                "Query reformulated: '%s' -> '%s'",
                original_query[:50],
                reformulated[:50],
            )
            return reformulated
        except Exception:
            logger.warning("Query reformulation failed, using original")
            return original_query

    def _generate_answer(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
    ) -> AnswerWithCitations:
        """Generate an answer with citations from context.

        Args:
            query: The user query.
            context_chunks: Retrieved and validated context.

        Returns:
            AnswerWithCitations with structured response.
        """
        if not context_chunks:
            return AnswerWithCitations(
                answer="I could not find sufficient information in the available "
                       "SEC filings to answer this question confidently.",
                citations=[],
                confidence=0.0,
            )

        context_text = "\n\n---\n\n".join(
            f"[Source: {c['citation']['source_document']}, "
            f"Section: {c['citation']['section']}]\n{c['content']}"
            for c in context_chunks
        )

        prompt = (
            f"{FINANCIAL_SYSTEM_PROMPT}\n\n"
            f"Context from SEC filings:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Provide a thorough answer based ONLY on the context above. "
            f"Cite the source document and section for each claim.\n"
            f"- If the context does not contain specific information to answer "
            f"any part of the question, explicitly state that the information "
            f"is not available rather than speculating.\n"
            f"- Do not make inferences or calculations beyond what is directly "
            f"stated in the context.\n"
            f"- Every numerical claim must reference the specific source."
        )

        query_type = self._detect_query_type(query)

        if query_type == "numerical":
            prompt += (
                "\n\nFORMAT INSTRUCTION: This is a numerical question. "
                "Lead with the specific number in your first sentence. "
                "State the exact figure with its unit (e.g., '$391.0 billion'). "
                "Keep the answer concise — 1-3 sentences maximum."
            )
        elif query_type == "comparative":
            prompt += (
                "\n\nFORMAT INSTRUCTION: This is a comparative question. "
                "Present each entity's value clearly, then state the difference or trend. "
                "Use a direct comparison format: 'X was $A while Y was $B, a difference of $C.' "
                "Include percentage changes where relevant. Keep the answer focused and structured."
            )
        else:  # analytical
            prompt += (
                "\n\nFORMAT INSTRUCTION: This is an analytical question. "
                "Provide a thorough but focused answer. Organize key points clearly. "
                "Cite specific sections for each major claim."
            )

        try:
            response = self._llm.complete(prompt)
            answer_text = response.text.strip()
            logger.info("LLM answer generated for: %s", query[:50])
        except Exception:
            logger.exception("LLM generation failed")
            return AnswerWithCitations(
                answer="An error occurred while generating the answer.",
                citations=[],
                confidence=0.0,
            )

        # Build citations from the context chunks used
        citations = [
            Citation(**c["citation"])
            for c in context_chunks
            if "citation" in c
        ]

        answer_obj = AnswerWithCitations(
            answer=answer_text,
            citations=citations,
            confidence=min(
                1.0,
                sum(c.get("score", 0.5) for c in context_chunks) / len(context_chunks),
            ),
        )
        answer_obj.contexts_used = context_chunks
        return answer_obj

    def _verify_grounding(
        self,
        answer_text: str,
        context_chunks: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """Verify that the generated answer is grounded in the provided context.

        Args:
            answer_text: The generated answer to verify.
            context_chunks: The context chunks used for generation.

        Returns:
            Tuple of (verified_answer, grounding_score).
            If grounding is poor, returns a revised answer with only supported claims.
        """
        context_text = "\n\n".join(
            c["content"][:2000] for c in context_chunks[:5]
        )

        prompt = (
            "You are a fact-checking assistant. Your task is to verify that every claim "
            "in the answer below is directly supported by the provided context.\n\n"
            "Context from SEC filings:\n"
            f"{context_text}\n\n"
            "Answer to verify:\n"
            f"{answer_text}\n\n"
            "Instructions:\n"
            "1. Check each factual claim in the answer against the context.\n"
            "2. A claim is SUPPORTED if the context contains the same information.\n"
            "3. A claim is UNSUPPORTED if the context does not contain that information.\n"
            "4. Score the overall grounding from 0.0 to 1.0 (fraction of supported claims).\n"
            "5. If any claims are unsupported, rewrite the answer including ONLY supported claims.\n\n"
            "Respond in this EXACT format (two lines only):\n"
            "SCORE: <decimal between 0.0 and 1.0>\n"
            "ANSWER: <the verified answer text>"
        )

        try:
            response = self._llm.complete(prompt)
            text = response.text.strip()
            # Extract score
            score_match = re.search(r'SCORE:\s*([\d.]+)', text)
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5

            # Extract answer
            answer_match = re.search(r'ANSWER:\s*(.+)', text, re.DOTALL)
            if answer_match:
                verified_answer = answer_match.group(1).strip()
            else:
                verified_answer = answer_text

            logger.info("Grounding verification score: %.2f", score)
            return verified_answer, score
        except Exception:
            logger.warning("Grounding verification failed, using fallback")
            return answer_text, 0.5

    def query(
        self,
        question: str,
        ticker: str | None = None,
    ) -> AnswerWithCitations:
        """Answer a question using CRAG: retrieve, evaluate, optionally re-retrieve.

        Args:
            question: User's question about financial filings.
            ticker: Optional ticker filter.

        Returns:
            AnswerWithCitations with answer and source citations.
        """
        logger.info("Query received: %s", question[:80])

        # Auto-extract ticker from query if not explicitly provided
        if ticker is None:
            ticker = self._extract_ticker(question)
            if ticker:
                logger.info("Auto-extracted ticker filter: %s", ticker)

        # Step 1: Initial retrieval
        context = self._retrieve_context(question, ticker=ticker)

        # Step 2: Evaluate context relevance (CRAG)
        confidence = self._evaluate_context_relevance(question, context)

        # Step 3: If below threshold, reformulate and re-retrieve
        if confidence < CRAG_CONFIDENCE_THRESHOLD:
            logger.info(
                "Low confidence (%.2f), reformulating query...", confidence
            )
            reformulated = self._reformulate_query(question)
            new_context = self._retrieve_context(reformulated, ticker=ticker)
            new_confidence = self._evaluate_context_relevance(reformulated, new_context)
            # Keep whichever retrieval scored higher
            if new_confidence > confidence:
                logger.info("Using reformulated results (%.2f > %.2f)", new_confidence, confidence)
                context = new_context
                confidence = new_confidence
            else:
                logger.info("Keeping original results (%.2f >= %.2f)", confidence, new_confidence)

        # Step 4: Generate answer
        answer = self._generate_answer(question, context)

        # Step 5: Verify grounding
        if context and answer.confidence > 0.0:
            verified_answer, grounding_score = self._verify_grounding(
                answer.answer, context,
            )
            if grounding_score < 0.4:
                answer = AnswerWithCitations(
                    answer="I could not find sufficient information in the available "
                           "SEC filings to answer this question confidently.",
                    citations=[],
                    confidence=0.0,
                    contexts_used=context,
                )
            else:
                answer.answer = verified_answer
                answer.confidence = min(answer.confidence, grounding_score)

        return answer

    def query_with_filters(
        self,
        question: str,
        ticker: str | None = None,
        filing_type: str | None = None,
    ) -> QueryResponse:
        """Answer a question with optional metadata filters.

        Args:
            question: User's question about financial filings.
            ticker: Filter results to a specific company ticker.
            filing_type: Filter results to a specific filing type.

        Returns:
            Structured QueryResponse with answer and source attributions.
        """
        result = self.query(question, ticker=ticker)

        sources = [
            SourceAttribution(
                chunk_id=f"{c.ticker}-{c.year}-{c.section}",
                text_excerpt=c.quote_snippet,
                filing_type=filing_type or "10-K",
                company=c.ticker,
                section=c.section,
            )
            for c in result.citations
        ]

        return QueryResponse(
            query=question,
            answer=result.answer,
            sources=sources,
            confidence=result.confidence,
        )
