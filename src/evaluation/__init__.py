"""Evaluation package — RAGAS evaluation harness with financial-specific metrics."""

from src.evaluation.models import EvalQuestion, EvalReport, QuestionCategory, QuestionResult
from src.evaluation.ragas_eval import RAGASEvaluator
from src.evaluation.test_questions import get_test_questions, load_test_questions, save_test_questions

__all__ = [
    "EvalQuestion",
    "EvalReport",
    "QuestionCategory",
    "QuestionResult",
    "RAGASEvaluator",
    "get_test_questions",
    "load_test_questions",
    "save_test_questions",
]
