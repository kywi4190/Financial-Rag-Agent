"""Tests for the evaluation package (RAGAS evaluator and question generator)."""

import pytest


class TestRagasEvaluator:
    """Tests for RagasEvaluator."""

    def test_evaluate_single_returns_metrics(self) -> None:
        """Test that evaluate_single produces RAGAS metrics."""
        ...

    def test_evaluate_batch_aggregates_metrics(self) -> None:
        """Test that evaluate_batch computes aggregate metrics."""
        ...


class TestTestQuestionGenerator:
    """Tests for TestQuestionGenerator."""

    def test_generate_from_section_returns_test_cases(self) -> None:
        """Test that question generation produces TestCase objects."""
        ...

    def test_generate_from_filing_covers_all_sections(self) -> None:
        """Test that filing-level generation covers every section."""
        ...
