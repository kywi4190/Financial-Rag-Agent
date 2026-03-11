"""Pydantic models for the RAGAS evaluation harness.

Defines structured representations for evaluation questions,
per-question results, and aggregate evaluation reports.
"""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Category of evaluation question."""

    NUMERICAL = "numerical"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"


class EvalQuestion(BaseModel):
    """A single evaluation question with ground truth.

    Attributes:
        question: The test question text.
        ground_truth_answer: Expected correct answer.
        ground_truth_contexts: Filing sections containing the answer.
        category: Question category (numerical, comparative, analytical).
    """

    question: str
    ground_truth_answer: str
    ground_truth_contexts: list[str]
    category: QuestionCategory


class QuestionResult(BaseModel):
    """Result of evaluating a single question through the RAG pipeline.

    Attributes:
        question: The original test question.
        predicted_answer: Answer produced by the query engine.
        retrieved_contexts: Context passages retrieved by the pipeline.
        scores: Dictionary mapping metric names to scores (0-1).
    """

    question: str
    predicted_answer: str
    retrieved_contexts: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Aggregate evaluation report across all test questions.

    Attributes:
        overall_scores: Averaged scores across all questions.
        per_category_scores: Averaged scores grouped by question category.
        per_question_results: Individual results for each question.
        timestamp: When the evaluation was run.
    """

    overall_scores: dict[str, float] = Field(default_factory=dict)
    per_category_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_question_results: list[QuestionResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
