"""RAGAS evaluation harness for the RAG pipeline.

Runs the RAGAS evaluation framework against the financial RAG pipeline,
measuring faithfulness, answer relevancy, context precision, and context recall.
"""

from src.evaluation.models import EvalReport, EvalResult, TestCase


class RagasEvaluator:
    """Evaluates the RAG pipeline using the RAGAS framework.

    Args:
        llm_model: OpenAI model identifier for RAGAS evaluation LLM.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        """Initialize the RAGAS evaluator."""
        ...

    def evaluate_single(self, test_case: TestCase) -> EvalResult:
        """Evaluate a single test case through the RAG pipeline.

        Args:
            test_case: A TestCase with question and ground truth.

        Returns:
            EvalResult with generated answer, contexts, and metrics.
        """
        ...

    def evaluate_batch(self, test_cases: list[TestCase]) -> EvalReport:
        """Evaluate a batch of test cases and produce an aggregate report.

        Args:
            test_cases: List of TestCase objects to evaluate.

        Returns:
            EvalReport with individual results and aggregate metrics.
        """
        ...
