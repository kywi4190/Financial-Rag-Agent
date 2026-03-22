"""RAGAS evaluation harness for the financial RAG pipeline.

Evaluates the pipeline using standard RAGAS metrics (faithfulness,
context precision, context recall, answer relevancy) plus custom
financial-domain metrics (citation accuracy, numerical accuracy).
"""

import logging
import re
from collections import defaultdict

from src.agents.query_engine import FinancialQueryEngine
from src.evaluation.models import (
    EvalQuestion,
    EvalReport,
    QuestionCategory,
    QuestionResult,
)

logger = logging.getLogger(__name__)

_MULTIPLIERS: dict[str, float] = {
    "trillion": 1e12,
    "billion": 1e9,
    "million": 1e6,
    "thousand": 1e3,
}


class LlamaIndexEmbeddingAdapter:
    """Adapter wrapping a LlamaIndex embedding model for RAGAS compatibility.

    RAGAS expects a langchain-compatible embedding object with ``embed_query()``
    and ``embed_documents()`` methods. This adapter delegates to LlamaIndex's
    ``OpenAIEmbedding``, which exposes ``get_text_embedding()`` and
    ``get_text_embedding_batch()``.

    Args:
        model_name: OpenAI embedding model identifier.
    """

    def __init__(self, model_name: str = "text-embedding-3-large") -> None:
        """Initialize the adapter with a LlamaIndex OpenAIEmbedding."""
        from llama_index.embeddings.openai import OpenAIEmbedding

        self._embedding = OpenAIEmbedding(model=model_name)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: The query string to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        return self._embedding.get_text_embedding(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors, one per document.
        """
        return self._embedding.get_text_embedding_batch(texts)


class RAGASEvaluator:
    """Evaluates the RAG pipeline using RAGAS and custom financial metrics.

    Args:
        llm_model: OpenAI model identifier for RAGAS evaluation LLM.
        embedding_model: OpenAI embedding model identifier. If ``None``,
            reads from project settings via ``get_settings()``.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o",
        embedding_model: str | None = None,
    ) -> None:
        """Initialize the evaluator."""
        self._llm_model = llm_model
        if embedding_model is None:
            try:
                from src.config import get_settings

                embedding_model = get_settings().embedding_model
            except Exception:
                embedding_model = "text-embedding-3-large"
        self._embedding_model = embedding_model

    def evaluate(
        self,
        test_questions: list[EvalQuestion],
        query_engine: FinancialQueryEngine,
    ) -> EvalReport:
        """Run full evaluation: query engine + RAGAS + custom metrics.

        Args:
            test_questions: List of evaluation questions with ground truth.
            query_engine: Configured FinancialQueryEngine to evaluate.

        Returns:
            EvalReport with per-question and aggregate scores.
        """
        logger.info("Starting evaluation with %d questions", len(test_questions))

        # Step 1: Run each question through the query engine
        results: list[QuestionResult] = []
        for i, question in enumerate(test_questions):
            logger.info(
                "Evaluating question %d/%d: %s",
                i + 1,
                len(test_questions),
                question.question[:60],
            )
            result = self._run_single_question(question, query_engine)
            results.append(result)

        # Step 2: Compute RAGAS metrics
        self._compute_ragas_metrics(test_questions, results)

        # Step 3: Compute custom financial metrics
        self._compute_custom_metrics(test_questions, results)

        # Step 4: Aggregate into report
        report = self._build_report(test_questions, results)
        logger.info("Evaluation complete. Overall scores: %s", report.overall_scores)
        return report

    def _run_single_question(
        self,
        question: EvalQuestion,
        query_engine: FinancialQueryEngine,
    ) -> QuestionResult:
        """Run a single question through the query engine.

        Args:
            question: The evaluation question.
            query_engine: The query engine to evaluate.

        Returns:
            QuestionResult with predicted answer and retrieved contexts.
        """
        try:
            answer = query_engine.query(question.question)
        except Exception:
            logger.exception("Query failed for: %s", question.question[:60])
            return QuestionResult(
                question=question.question,
                predicted_answer="Error: query engine failed.",
                retrieved_contexts=[],
                scores={},
            )

        # Collect retrieved contexts
        contexts: list[str] = []
        if query_engine._retriever is not None:
            try:
                raw_contexts = query_engine._retrieve_context(question.question)
                contexts = [
                    f"{c['citation']['source_document']}, {c['citation']['section']}: {c['content']}"
                    for c in raw_contexts
                ]
            except Exception:
                logger.warning("Context retrieval failed, using citation snippets")

        # Fall back to citation snippets if no full contexts
        if not contexts and answer.citations:
            contexts = [
                f"{c.source_document}, {c.section}: {c.quote_snippet}"
                for c in answer.citations
            ]

        return QuestionResult(
            question=question.question,
            predicted_answer=answer.answer,
            retrieved_contexts=contexts,
            scores={},
        )

    def _compute_ragas_metrics(
        self,
        questions: list[EvalQuestion],
        results: list[QuestionResult],
    ) -> None:
        """Compute standard RAGAS metrics using the ragas library.

        Scores are written directly into each QuestionResult.scores dict.

        Args:
            questions: Evaluation questions with ground truth.
            results: Corresponding query results to score.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:
            logger.warning(
                "ragas or datasets not installed. "
                "Skipping RAGAS metrics (install with: pip install ragas datasets)."
            )
            for result in results:
                for name in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]:
                    result.scores[name] = 0.0
            return

        try:
            # Build RAGAS dataset using ragas 0.4.x column names
            dataset = Dataset.from_dict(
                {
                    "user_input": [q.question for q in questions],
                    "response": [r.predicted_answer for r in results],
                    "retrieved_contexts": [r.retrieved_contexts for r in results],
                    "reference": [q.ground_truth_answer for q in questions],
                }
            )

            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            embedding_adapter = LlamaIndexEmbeddingAdapter(
                model_name=self._embedding_model,
            )
            ragas_result = ragas_evaluate(
                dataset, metrics=metrics, embeddings=embedding_adapter,
            )

            # Extract per-question scores from the result DataFrame
            try:
                df = ragas_result.to_pandas()
                metric_names = [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
                for i, result in enumerate(results):
                    for name in metric_names:
                        if name in df.columns and i < len(df):
                            score = df.iloc[i][name]
                            result.scores[name] = float(score) if score == score else 0.0
                        else:
                            result.scores[name] = 0.0
            except Exception:
                # Fall back to aggregate scores applied uniformly
                logger.warning("Could not extract per-row RAGAS scores, using aggregates")
                for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                    avg = float(ragas_result.get(name, 0.0))
                    for result in results:
                        result.scores[name] = avg

        except Exception:
            logger.exception("RAGAS evaluation failed")
            for result in results:
                for name in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]:
                    result.scores.setdefault(name, 0.0)

    def _compute_custom_metrics(
        self,
        questions: list[EvalQuestion],
        results: list[QuestionResult],
    ) -> None:
        """Compute custom financial-domain metrics.

        Computes citation_accuracy for all questions and
        numerical_accuracy for numerical category questions.

        Args:
            questions: Evaluation questions with ground truth.
            results: Corresponding query results to score.
        """
        for question, result in zip(questions, results):
            result.scores["citation_accuracy"] = self._compute_citation_accuracy(
                question, result
            )
            if question.category == QuestionCategory.NUMERICAL:
                result.scores["numerical_accuracy"] = (
                    self._compute_numerical_accuracy(question, result)
                )

    @staticmethod
    def _parse_ground_truth_context(gt_context: str) -> dict[str, str] | None:
        """Parse a ground truth context string into components.

        Expected format: ``"{ticker} {filing_type} {year}, {section}"``
        e.g. ``"AAPL 10-K 2024, Item 8. Financial Statements"``.

        Args:
            gt_context: Ground truth context string.

        Returns:
            Dict with keys ticker, filing_type, year, section, or None if
            the format doesn't match.
        """
        match = re.match(
            r"^([A-Z]{1,5})\s+(10-[KQ]|20-F|8-K)\s+(\d{4}),\s+(.+)$",
            gt_context.strip(),
        )
        if not match:
            return None
        return {
            "ticker": match.group(1),
            "filing_type": match.group(2),
            "year": match.group(3),
            "section": match.group(4),
        }

    @staticmethod
    def _compute_citation_accuracy(
        question: EvalQuestion,
        result: QuestionResult,
    ) -> float:
        """Compute what fraction of retrieved contexts match ground truth sections.

        Uses component-based matching when the ground truth format is parseable,
        falling back to substring matching otherwise.

        Scoring per retrieved context (best match across ground truths):
        - ticker + section + year all match → 1.0
        - section + year match (wrong/missing ticker) → 0.5
        - section only matches → 0.25
        - no section match → 0.0

        Args:
            question: Question with ground truth context sections.
            result: Result with retrieved contexts.

        Returns:
            Score between 0.0 and 1.0.
        """
        if not result.retrieved_contexts:
            return 0.0

        parsed_gts = [
            RAGASEvaluator._parse_ground_truth_context(gt)
            for gt in question.ground_truth_contexts
        ]

        total_score = 0.0
        for ctx in result.retrieved_contexts:
            ctx_lower = ctx.lower()
            best_score = 0.0

            for gt_raw, parsed in zip(question.ground_truth_contexts, parsed_gts):
                if parsed is not None:
                    # Component-based matching
                    section_match = parsed["section"].lower() in ctx_lower
                    year_match = parsed["year"] in ctx
                    ticker = parsed["ticker"]
                    # Word boundary or bracket format match
                    ticker_match = bool(
                        re.search(
                            rf"(?<![A-Za-z]){re.escape(ticker)}(?![A-Za-z])",
                            ctx,
                            re.IGNORECASE,
                        )
                    )

                    if section_match and year_match and ticker_match:
                        score = 1.0
                    elif section_match and year_match:
                        score = 0.5
                    elif section_match:
                        score = 0.25
                    else:
                        score = 0.0
                else:
                    # Fallback: original substring matching
                    if gt_raw.lower() in ctx_lower:
                        score = 1.0
                    else:
                        section_parts = gt_raw.split(", ")
                        if len(section_parts) >= 2 and section_parts[-1].lower() in ctx_lower:
                            score = 1.0
                        else:
                            score = 0.0

                best_score = max(best_score, score)

            total_score += best_score

        return total_score / len(result.retrieved_contexts)

    @staticmethod
    def _compute_numerical_accuracy(
        question: EvalQuestion,
        result: QuestionResult,
    ) -> float:
        """Check if extracted numbers from predicted answer match ground truth.

        Uses 1% tolerance for numerical comparison.

        Args:
            question: Question with ground truth answer containing numbers.
            result: Result with predicted answer.

        Returns:
            Score between 0.0 and 1.0.
        """
        gt_numbers = RAGASEvaluator._extract_numbers(question.ground_truth_answer)
        if not gt_numbers:
            return 1.0

        pred_numbers = RAGASEvaluator._extract_numbers(result.predicted_answer)
        if not pred_numbers:
            return 0.0

        matches = 0
        for gt_num in gt_numbers:
            for pred_num in pred_numbers:
                if gt_num == 0.0:
                    if pred_num == 0.0:
                        matches += 1
                        break
                elif abs(pred_num - gt_num) / abs(gt_num) <= 0.01:
                    matches += 1
                    break

        return matches / len(gt_numbers)

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract numeric values from text, handling financial formatting.

        Handles dollar signs, commas, and multiplier words
        (million, billion, trillion).

        Args:
            text: Text potentially containing financial numbers.

        Returns:
            List of extracted numeric values.
        """
        cleaned = text.replace("$", "").replace(",", "")

        numbers: list[float] = []
        # Match numbers optionally followed by a multiplier word
        pattern = r"(-?\d+\.?\d*)\s*(trillion|billion|million|thousand)?"
        for match in re.finditer(pattern, cleaned, re.IGNORECASE):
            val = float(match.group(1))
            suffix = match.group(2)
            if suffix:
                multiplier = _MULTIPLIERS.get(suffix.lower())
                if multiplier:
                    val *= multiplier
            numbers.append(val)

        return numbers

    @staticmethod
    def _build_report(
        questions: list[EvalQuestion],
        results: list[QuestionResult],
    ) -> EvalReport:
        """Aggregate per-question scores into an EvalReport.

        Args:
            questions: Evaluation questions (for category grouping).
            results: Scored question results.

        Returns:
            EvalReport with overall and per-category scores.
        """
        # Collect all metric names
        all_metrics: set[str] = set()
        for r in results:
            all_metrics.update(r.scores.keys())

        # Overall scores: mean across all questions
        overall: dict[str, float] = {}
        for metric in sorted(all_metrics):
            scores = [r.scores[metric] for r in results if metric in r.scores]
            if scores:
                overall[metric] = sum(scores) / len(scores)

        # Per-category scores
        category_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for question, result in zip(questions, results):
            cat = question.category.value
            for metric, score in result.scores.items():
                category_scores[cat][metric].append(score)

        per_category: dict[str, dict[str, float]] = {}
        for cat, metrics in category_scores.items():
            per_category[cat] = {
                m: sum(scores) / len(scores) for m, scores in sorted(metrics.items())
            }

        return EvalReport(
            overall_scores=overall,
            per_category_scores=per_category,
            per_question_results=results,
        )
