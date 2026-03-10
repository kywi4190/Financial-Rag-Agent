"""CLI script to ingest SEC filings into the vector store.

Usage:
    python scripts/ingest.py --ticker AAPL --filing-type 10-K --num-filings 3

Downloads filings from SEC EDGAR, parses them into sections,
chunks the content, and indexes chunks into ChromaDB.
"""

import argparse


def main() -> None:
    """Run the ingestion pipeline from the command line."""
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into the vector store."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--filing-type", default="10-K", help="SEC filing type (default: 10-K)"
    )
    parser.add_argument(
        "--num-filings",
        type=int,
        default=3,
        help="Number of filings to ingest (default: 3)",
    )
    _args = parser.parse_args()

    # TODO: Implement ingestion pipeline
    # 1. Initialize EdgarClient with SEC_EDGAR_IDENTITY
    # 2. Download filings
    # 3. Parse into sections
    # 4. Chunk sections
    # 5. Index chunks into VectorStore and BM25Search


if __name__ == "__main__":
    main()
