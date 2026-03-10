"""LlamaIndex query engines for financial document Q&A.

Wraps LlamaIndex query engine functionality with the project's
retrieval pipeline to answer questions about SEC filings.
"""

from src.agents.models import QueryResponse


class FinancialQueryEngine:
    """Query engine for answering questions about financial filings.

    Integrates the hybrid retrieval pipeline with LlamaIndex's
    query engine for response synthesis.

    Args:
        llm_model: OpenAI model identifier for generation.
        embedding_model: OpenAI model identifier for embeddings.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the query engine with LLM and embedding models."""
        ...

    def query(self, question: str) -> QueryResponse:
        """Answer a natural language question using retrieved context.

        Args:
            question: User's question about financial filings.

        Returns:
            Structured QueryResponse with answer and source attributions.
        """
        ...

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
        ...
