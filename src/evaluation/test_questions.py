"""Synthetic Q&A generation for evaluation.

Generates test questions and ground-truth answers from financial
filings to create evaluation datasets for the RAG pipeline.
"""

from src.evaluation.models import TestCase
from src.ingestion.models import FilingSection


class TestQuestionGenerator:
    """Generates synthetic test Q&A pairs from filing sections.

    Args:
        llm_model: OpenAI model identifier for question generation.
        questions_per_section: Number of questions to generate per section.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        questions_per_section: int = 3,
    ) -> None:
        """Initialize the question generator."""
        ...

    def generate_from_section(self, section: FilingSection) -> list[TestCase]:
        """Generate test questions from a single filing section.

        Args:
            section: A parsed filing section.

        Returns:
            List of TestCase objects with questions and ground truths.
        """
        ...

    def generate_from_filing(
        self, sections: list[FilingSection]
    ) -> list[TestCase]:
        """Generate test questions from all sections of a filing.

        Args:
            sections: All parsed sections from a single filing.

        Returns:
            Complete list of TestCase objects for the filing.
        """
        ...
