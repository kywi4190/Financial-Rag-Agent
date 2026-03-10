"""Pydantic models for evaluation results.

Defines structured representations of evaluation metrics, individual
test case results, and aggregate evaluation reports.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single evaluation test case.

    Attributes:
        question: The test question.
        ground_truth: Expected answer or key facts.
        contexts: List of ground-truth context passages.
        metadata: Additional metadata (e.g., difficulty, category).
    """

    question: str
    ground_truth: str
    contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class EvalMetrics(BaseModel):
    """RAGAS evaluation metrics for a single test case.

    Attributes:
        faithfulness: Score measuring answer grounding in context (0-1).
        answer_relevancy: Score measuring answer relevance to question (0-1).
        context_precision: Score measuring context ranking quality (0-1).
        context_recall: Score measuring context coverage of ground truth (0-1).
    """

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


class EvalResult(BaseModel):
    """Result of evaluating a single test case.

    Attributes:
        test_case: The test case that was evaluated.
        generated_answer: The answer produced by the RAG pipeline.
        retrieved_contexts: Contexts retrieved by the pipeline.
        metrics: RAGAS metrics for this test case.
    """

    test_case: TestCase
    generated_answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    metrics: EvalMetrics = Field(default_factory=EvalMetrics)


class EvalReport(BaseModel):
    """Aggregate evaluation report across all test cases.

    Attributes:
        results: Individual evaluation results.
        aggregate_metrics: Averaged metrics across all test cases.
        timestamp: When the evaluation was run.
        config: Configuration snapshot used for the evaluation.
    """

    results: list[EvalResult] = Field(default_factory=list)
    aggregate_metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: dict[str, str] = Field(default_factory=dict)
