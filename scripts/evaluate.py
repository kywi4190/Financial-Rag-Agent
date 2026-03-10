"""CLI script to run RAGAS evaluation on the RAG pipeline.

Usage:
    python scripts/evaluate.py --ticker AAPL --num-questions 10

Generates synthetic test questions from ingested filings and
evaluates the RAG pipeline using RAGAS metrics.
"""

import argparse


def main() -> None:
    """Run the evaluation pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG pipeline using RAGAS."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=10,
        help="Number of test questions to generate (default: 10)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the evaluation report JSON",
    )
    _args = parser.parse_args()

    # TODO: Implement evaluation pipeline
    # 1. Generate synthetic test questions
    # 2. Run RAGAS evaluation
    # 3. Print/save aggregate metrics


if __name__ == "__main__":
    main()
