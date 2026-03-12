"""CLI to run RAGAS evaluation and print formatted results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --questions data/eval/test_questions.json
    python scripts/evaluate.py --output data/eval/custom_results.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agents.query_engine import FinancialQueryEngine
from src.chunking.models import ChunkMetadata, DocumentChunk
from src.evaluation.models import EvalReport
from src.evaluation.ragas_eval import RAGASEvaluator
from src.evaluation.test_questions import (
    DEFAULT_QUESTIONS_PATH,
    get_test_questions,
    load_test_questions,
    save_test_questions,
)
from src.config import get_settings
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalConfig
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/eval")


def print_results_table(report: EvalReport) -> None:
    """Print a formatted summary of evaluation results.

    Args:
        report: The evaluation report to display.
    """
    print("\n" + "=" * 70)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Questions evaluated: {len(report.per_question_results)}")

    # Overall scores
    print("\n" + "-" * 70)
    print("  OVERALL SCORES")
    print("-" * 70)
    print(f"  {'Metric':<25} {'Score':>10}")
    print(f"  {'-' * 25} {'-' * 10}")
    for metric, score in sorted(report.overall_scores.items()):
        print(f"  {metric:<25} {score:>10.4f}")

    # Per-category scores
    print("\n" + "-" * 70)
    print("  SCORES BY CATEGORY")
    print("-" * 70)
    for category, scores in sorted(report.per_category_scores.items()):
        print(f"\n  [{category.upper()}]")
        print(f"  {'Metric':<25} {'Score':>10}")
        print(f"  {'-' * 25} {'-' * 10}")
        for metric, score in sorted(scores.items()):
            print(f"  {metric:<25} {score:>10.4f}")

    # Per-question summary (abbreviated)
    print("\n" + "-" * 70)
    print("  PER-QUESTION RESULTS (first 10)")
    print("-" * 70)
    for i, result in enumerate(report.per_question_results[:10]):
        q_short = result.question[:55] + "..." if len(result.question) > 55 else result.question
        avg_score = (
            sum(result.scores.values()) / len(result.scores)
            if result.scores
            else 0.0
        )
        print(f"  {i + 1:>3}. [{avg_score:.3f}] {q_short}")

    if len(report.per_question_results) > 10:
        print(f"  ... and {len(report.per_question_results) - 10} more questions")

    print("\n" + "=" * 70)


def save_results(report: EvalReport, output_path: Path | None = None) -> Path:
    """Save evaluation results to a timestamped JSON file.

    Args:
        report: The evaluation report to save.
        output_path: Optional explicit output path.

    Returns:
        Path where results were saved.
    """
    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"results_{ts}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Run the evaluation pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline using RAGAS metrics."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help=f"Path to test questions JSON (default: {DEFAULT_QUESTIONS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results JSON (default: data/eval/results_{timestamp}.json)",
    )
    parser.add_argument(
        "--generate-questions",
        action="store_true",
        help="Generate and save test questions JSON, then exit",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for RAGAS evaluation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--rebuild-bm25",
        action="store_true",
        help="Force rebuild BM25 index from ChromaDB instead of loading from disk",
    )
    args = parser.parse_args()

    # Generate-only mode
    if args.generate_questions:
        path = save_test_questions(path=args.questions)
        print(f"Saved test questions to {path}")
        return

    # Load or generate test questions
    questions_path = args.questions or DEFAULT_QUESTIONS_PATH
    if questions_path.exists():
        logger.info("Loading test questions from %s", questions_path)
        test_questions = load_test_questions(questions_path)
    else:
        logger.info("No questions file found, using built-in questions")
        test_questions = get_test_questions()
        save_test_questions(test_questions, questions_path)
        logger.info("Saved questions to %s", questions_path)

    logger.info(
        "Loaded %d test questions (%d numerical, %d comparative, %d analytical)",
        len(test_questions),
        sum(1 for q in test_questions if q.category.value == "numerical"),
        sum(1 for q in test_questions if q.category.value == "comparative"),
        sum(1 for q in test_questions if q.category.value == "analytical"),
    )

    # Set up retriever pipeline and query engine
    try:
        vector_store = ChromaStore()
        stats = vector_store.get_collection_stats()
        if stats.get("count", 0) == 0:
            logger.error("No documents indexed. Run scripts/ingest.py first.")
            sys.exit(1)
        logger.info(
            "ChromaDB: %d chunks, tickers=%s, years=%s",
            stats["count"], stats["tickers"], stats["years"],
        )

        bm25_path = Path(get_settings().bm25_persist_dir) / "bm25_index.pkl"
        if bm25_path.exists() and not args.rebuild_bm25:
            logger.info("Loading BM25 index from %s", bm25_path)
            bm25 = BM25Index.load_index(bm25_path)
        else:
            bm25 = BM25Index()
            all_docs = vector_store._collection.get(include=["documents", "metadatas"])
            if all_docs["documents"]:
                chunks = []
                for i, doc in enumerate(all_docs["documents"]):
                    meta = all_docs["metadatas"][i]
                    chunks.append(DocumentChunk(
                        chunk_id=all_docs["ids"][i],
                        content=doc,
                        metadata=ChunkMetadata(**meta),
                    ))
                bm25.build_index(chunks)
                bm25.save_index(bm25_path)

        reranker = Reranker()
        retriever = HybridRetriever(vector_store, bm25, RetrievalConfig(), reranker=reranker)
        query_engine = FinancialQueryEngine(retriever=retriever)
    except Exception:
        logger.exception("Failed to initialize query engine")
        sys.exit(1)

    # Run evaluation
    evaluator = RAGASEvaluator(llm_model=args.llm_model)
    report = evaluator.evaluate(test_questions, query_engine)

    # Save results first (before printing, in case display fails)
    output_path = save_results(report, args.output)
    print(f"\nResults saved to: {output_path}")

    # Print results
    print_results_table(report)


if __name__ == "__main__":
    main()
