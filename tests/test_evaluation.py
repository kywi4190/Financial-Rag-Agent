"""Tests for the evaluation package (models, test questions, RAGAS evaluator)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.models import (
    EvalQuestion,
    EvalReport,
    QuestionCategory,
    QuestionResult,
)
from src.evaluation.ragas_eval import RAGASEvaluator
from src.evaluation.test_questions import (
    get_test_questions,
    load_test_questions,
    save_test_questions,
)


# ── Model Tests ────────────────────────────────────────────────────────────


class TestEvalQuestion:
    """Tests for the EvalQuestion model."""

    def test_creation(self) -> None:
        q = EvalQuestion(
            question="What was revenue?",
            ground_truth_answer="Revenue was $100 billion.",
            ground_truth_contexts=["AAPL 10-K 2024, Item 8. Financial Statements"],
            category=QuestionCategory.NUMERICAL,
        )
        assert q.question == "What was revenue?"
        assert q.category == QuestionCategory.NUMERICAL
        assert len(q.ground_truth_contexts) == 1

    def test_category_values(self) -> None:
        assert QuestionCategory.NUMERICAL.value == "numerical"
        assert QuestionCategory.COMPARATIVE.value == "comparative"
        assert QuestionCategory.ANALYTICAL.value == "analytical"


class TestQuestionResult:
    """Tests for the QuestionResult model."""

    def test_creation_with_scores(self) -> None:
        r = QuestionResult(
            question="What was revenue?",
            predicted_answer="Revenue was $100B.",
            retrieved_contexts=["Context about revenue."],
            scores={"faithfulness": 0.9, "numerical_accuracy": 1.0},
        )
        assert r.scores["faithfulness"] == 0.9
        assert len(r.retrieved_contexts) == 1

    def test_defaults(self) -> None:
        r = QuestionResult(question="q", predicted_answer="a")
        assert r.retrieved_contexts == []
        assert r.scores == {}


class TestEvalReport:
    """Tests for the EvalReport model."""

    def test_creation(self) -> None:
        report = EvalReport(
            overall_scores={"faithfulness": 0.85},
            per_category_scores={"numerical": {"faithfulness": 0.9}},
            per_question_results=[],
        )
        assert report.overall_scores["faithfulness"] == 0.85
        assert report.timestamp is not None

    def test_serialization_roundtrip(self) -> None:
        report = EvalReport(
            overall_scores={"faithfulness": 0.85, "citation_accuracy": 0.7},
            per_category_scores={
                "numerical": {"faithfulness": 0.9, "numerical_accuracy": 0.8}
            },
            per_question_results=[
                QuestionResult(question="q1", predicted_answer="a1", scores={"f": 0.9})
            ],
        )
        json_str = report.model_dump_json()
        restored = EvalReport.model_validate_json(json_str)
        assert restored.overall_scores == report.overall_scores
        assert len(restored.per_question_results) == 1


# ── Test Questions Tests ───────────────────────────────────────────────────


class TestTestQuestions:
    """Tests for the test question generation and I/O."""

    def test_get_test_questions_returns_50(self) -> None:
        questions = get_test_questions()
        assert len(questions) == 50

    def test_category_distribution(self) -> None:
        questions = get_test_questions()
        numerical = [q for q in questions if q.category == QuestionCategory.NUMERICAL]
        comparative = [q for q in questions if q.category == QuestionCategory.COMPARATIVE]
        analytical = [q for q in questions if q.category == QuestionCategory.ANALYTICAL]
        assert len(numerical) == 20
        assert len(comparative) == 15
        assert len(analytical) == 15

    def test_all_questions_have_ground_truth(self) -> None:
        questions = get_test_questions()
        for q in questions:
            assert q.ground_truth_answer, f"Missing ground truth for: {q.question}"
            assert q.ground_truth_contexts, f"Missing contexts for: {q.question}"

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        questions = get_test_questions()
        path = tmp_path / "test_questions.json"
        save_test_questions(questions, path)

        assert path.exists()
        loaded = load_test_questions(path)
        assert len(loaded) == len(questions)
        assert loaded[0].question == questions[0].question
        assert loaded[0].category == questions[0].category

    def test_save_creates_json(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "questions.json"
        save_test_questions(path=path)

        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 50
        assert "question" in data[0]
        assert "ground_truth_answer" in data[0]
        assert "category" in data[0]

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_test_questions(tmp_path / "nonexistent.json")


# ── Number Extraction Tests ────────────────────────────────────────────────


class TestExtractNumbers:
    """Tests for RAGASEvaluator._extract_numbers."""

    def test_plain_numbers(self) -> None:
        nums = RAGASEvaluator._extract_numbers("Revenue was 100 and profit was 50.")
        assert 100.0 in nums
        assert 50.0 in nums

    def test_dollar_signs_and_commas(self) -> None:
        nums = RAGASEvaluator._extract_numbers("Total was $383,285,000,000.")
        assert 383285000000.0 in nums

    def test_billion_suffix(self) -> None:
        nums = RAGASEvaluator._extract_numbers("Revenue was $391.0 billion.")
        assert any(abs(n - 391e9) < 1e6 for n in nums)

    def test_million_suffix(self) -> None:
        nums = RAGASEvaluator._extract_numbers("Costs were $210.35 million.")
        assert any(abs(n - 210.35e6) < 1e3 for n in nums)

    def test_percentages(self) -> None:
        nums = RAGASEvaluator._extract_numbers("Margin was 46.2%.")
        assert 46.2 in nums

    def test_mixed_formats(self) -> None:
        text = "Revenue of $391.0 billion with a 46.2% margin."
        nums = RAGASEvaluator._extract_numbers(text)
        assert any(abs(n - 391e9) < 1e6 for n in nums)
        assert 46.2 in nums

    def test_empty_string(self) -> None:
        assert RAGASEvaluator._extract_numbers("") == []

    def test_no_numbers(self) -> None:
        assert RAGASEvaluator._extract_numbers("No numbers here.") == []


# ── Numerical Accuracy Tests ──────────────────────────────────────────────


class TestNumericalAccuracy:
    """Tests for RAGASEvaluator._compute_numerical_accuracy."""

    def test_exact_match(self) -> None:
        q = EvalQuestion(
            question="Revenue?",
            ground_truth_answer="Revenue was $391.0 billion.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="Revenue?",
            predicted_answer="Revenue was $391.0 billion.",
        )
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 1.0

    def test_within_tolerance(self) -> None:
        q = EvalQuestion(
            question="Revenue?",
            ground_truth_answer="Revenue was $391.0 billion.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="Revenue?",
            predicted_answer="Revenue was $393.0 billion.",  # ~0.5% off
        )
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 1.0

    def test_outside_tolerance(self) -> None:
        q = EvalQuestion(
            question="Revenue?",
            ground_truth_answer="Revenue was $391.0 billion.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="Revenue?",
            predicted_answer="Revenue was $350.0 billion.",  # ~10% off
        )
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 0.0

    def test_no_numbers_in_ground_truth(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="No numbers here.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(question="q", predicted_answer="answer 123")
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 1.0

    def test_no_numbers_in_prediction(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="Revenue was $100 billion.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(question="q", predicted_answer="I don't know.")
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 0.0

    def test_multiple_numbers_partial_match(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="Revenue $391.0 billion, income $93.7 billion.",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="q",
            predicted_answer="Revenue $391.0 billion, income $80.0 billion.",
        )
        score = RAGASEvaluator._compute_numerical_accuracy(q, r)
        assert score == 0.5  # 1 of 2 numbers match


# ── Citation Accuracy Tests ───────────────────────────────────────────────


class TestCitationAccuracy:
    """Tests for RAGASEvaluator._compute_citation_accuracy."""

    def test_full_match(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="a",
            ground_truth_contexts=["AAPL 10-K 2024, Item 8. Financial Statements"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="q",
            predicted_answer="a",
            retrieved_contexts=[
                "AAPL 10-K 2024, Item 8. Financial Statements: Revenue was..."
            ],
        )
        score = RAGASEvaluator._compute_citation_accuracy(q, r)
        assert score == 1.0

    def test_partial_section_match(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="a",
            ground_truth_contexts=["AAPL 10-K 2024, Item 7. MD&A"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="q",
            predicted_answer="a",
            retrieved_contexts=[
                "From Item 7. MD&A: revenue discussion...",
            ],
        )
        score = RAGASEvaluator._compute_citation_accuracy(q, r)
        assert score == 1.0

    def test_no_match(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="a",
            ground_truth_contexts=["AAPL 10-K 2024, Item 8. Financial Statements"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="q",
            predicted_answer="a",
            retrieved_contexts=[
                "From Item 1A. Risk Factors: Some unrelated text...",
            ],
        )
        score = RAGASEvaluator._compute_citation_accuracy(q, r)
        assert score == 0.0

    def test_empty_contexts(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="a",
            ground_truth_contexts=["ctx"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(question="q", predicted_answer="a", retrieved_contexts=[])
        score = RAGASEvaluator._compute_citation_accuracy(q, r)
        assert score == 0.0

    def test_mixed_match(self) -> None:
        q = EvalQuestion(
            question="q",
            ground_truth_answer="a",
            ground_truth_contexts=["AAPL 10-K 2024, Item 7. MD&A"],
            category=QuestionCategory.NUMERICAL,
        )
        r = QuestionResult(
            question="q",
            predicted_answer="a",
            retrieved_contexts=[
                "Item 7. MD&A: Revenue grew...",
                "Item 1A. Risk Factors: Unrelated risk...",
            ],
        )
        score = RAGASEvaluator._compute_citation_accuracy(q, r)
        assert score == 0.5


# ── Report Aggregation Tests ──────────────────────────────────────────────


class TestBuildReport:
    """Tests for RAGASEvaluator._build_report."""

    def test_aggregates_overall_scores(self) -> None:
        questions = [
            EvalQuestion(
                question="q1",
                ground_truth_answer="a1",
                ground_truth_contexts=["c1"],
                category=QuestionCategory.NUMERICAL,
            ),
            EvalQuestion(
                question="q2",
                ground_truth_answer="a2",
                ground_truth_contexts=["c2"],
                category=QuestionCategory.NUMERICAL,
            ),
        ]
        results = [
            QuestionResult(
                question="q1",
                predicted_answer="a1",
                scores={"faithfulness": 0.8, "numerical_accuracy": 1.0},
            ),
            QuestionResult(
                question="q2",
                predicted_answer="a2",
                scores={"faithfulness": 0.6, "numerical_accuracy": 0.5},
            ),
        ]
        report = RAGASEvaluator._build_report(questions, results)
        assert report.overall_scores["faithfulness"] == pytest.approx(0.7)
        assert report.overall_scores["numerical_accuracy"] == pytest.approx(0.75)

    def test_groups_by_category(self) -> None:
        questions = [
            EvalQuestion(
                question="q1",
                ground_truth_answer="a",
                ground_truth_contexts=["c"],
                category=QuestionCategory.NUMERICAL,
            ),
            EvalQuestion(
                question="q2",
                ground_truth_answer="a",
                ground_truth_contexts=["c"],
                category=QuestionCategory.ANALYTICAL,
            ),
        ]
        results = [
            QuestionResult(
                question="q1",
                predicted_answer="a",
                scores={"faithfulness": 0.9},
            ),
            QuestionResult(
                question="q2",
                predicted_answer="a",
                scores={"faithfulness": 0.5},
            ),
        ]
        report = RAGASEvaluator._build_report(questions, results)
        assert "numerical" in report.per_category_scores
        assert "analytical" in report.per_category_scores
        assert report.per_category_scores["numerical"]["faithfulness"] == 0.9
        assert report.per_category_scores["analytical"]["faithfulness"] == 0.5


# ── Evaluator Integration Tests ──────────────────────────────────────────


class TestRAGASEvaluator:
    """Integration tests for RAGASEvaluator with mocked query engine."""

    @pytest.fixture()
    def mock_query_engine(self) -> MagicMock:
        """Create a mock FinancialQueryEngine."""
        engine = MagicMock()
        engine._retriever = None

        answer = MagicMock()
        answer.answer = "Apple's revenue was $391.0 billion in FY2024."
        answer.citations = []
        answer.confidence = 0.85
        engine.query.return_value = answer

        return engine

    @pytest.fixture()
    def sample_questions(self) -> list[EvalQuestion]:
        """Create a small set of test questions."""
        return [
            EvalQuestion(
                question="What was Apple's revenue in FY2024?",
                ground_truth_answer="Revenue was $391.0 billion.",
                ground_truth_contexts=["AAPL 10-K 2024, Item 8. Financial Statements"],
                category=QuestionCategory.NUMERICAL,
            ),
            EvalQuestion(
                question="What are Apple's key risk factors?",
                ground_truth_answer="Apple faces supply chain, regulatory, and competition risks.",
                ground_truth_contexts=["AAPL 10-K 2024, Item 1A. Risk Factors"],
                category=QuestionCategory.ANALYTICAL,
            ),
        ]

    def test_evaluate_returns_report(
        self,
        mock_query_engine: MagicMock,
        sample_questions: list[EvalQuestion],
    ) -> None:
        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(sample_questions, mock_query_engine)

        assert isinstance(report, EvalReport)
        assert len(report.per_question_results) == 2
        assert report.overall_scores  # Should have at least custom metrics
        assert report.timestamp is not None

    def test_evaluate_calls_query_engine(
        self,
        mock_query_engine: MagicMock,
        sample_questions: list[EvalQuestion],
    ) -> None:
        evaluator = RAGASEvaluator()
        evaluator.evaluate(sample_questions, mock_query_engine)
        assert mock_query_engine.query.call_count == 2

    def test_evaluate_computes_custom_metrics(
        self,
        mock_query_engine: MagicMock,
        sample_questions: list[EvalQuestion],
    ) -> None:
        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(sample_questions, mock_query_engine)

        # All questions should have citation_accuracy
        for r in report.per_question_results:
            assert "citation_accuracy" in r.scores

        # Numerical question should also have numerical_accuracy
        numerical_results = [
            r
            for r, q in zip(
                report.per_question_results, sample_questions
            )
            if q.category == QuestionCategory.NUMERICAL
        ]
        for r in numerical_results:
            assert "numerical_accuracy" in r.scores

    def test_evaluate_handles_query_failure(
        self,
        sample_questions: list[EvalQuestion],
    ) -> None:
        engine = MagicMock()
        engine._retriever = None
        engine.query.side_effect = RuntimeError("LLM failure")

        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(sample_questions, engine)

        assert len(report.per_question_results) == 2
        for r in report.per_question_results:
            assert "Error" in r.predicted_answer

    def test_evaluate_per_category_scores(
        self,
        mock_query_engine: MagicMock,
        sample_questions: list[EvalQuestion],
    ) -> None:
        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(sample_questions, mock_query_engine)

        assert "numerical" in report.per_category_scores
        assert "analytical" in report.per_category_scores

    def test_evaluate_with_retriever(
        self,
        sample_questions: list[EvalQuestion],
    ) -> None:
        """Test that evaluator uses retriever contexts when available."""
        engine = MagicMock()

        answer = MagicMock()
        answer.answer = "Revenue was $391.0 billion."
        answer.citations = []
        answer.confidence = 0.9
        engine.query.return_value = answer

        # Mock retriever that returns contexts
        engine._retriever = MagicMock()
        engine._retrieve_context.return_value = [
            {"content": "Item 8. Financial Statements: Revenue was $391B."},
        ]

        evaluator = RAGASEvaluator()
        report = evaluator.evaluate(sample_questions[:1], engine)

        result = report.per_question_results[0]
        assert len(result.retrieved_contexts) == 1
        assert "Financial Statements" in result.retrieved_contexts[0]
