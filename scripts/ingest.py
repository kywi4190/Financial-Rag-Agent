"""CLI script to ingest SEC filings and save parsed data as JSON.

Usage:
    python scripts/ingest.py --ticker AAPL
    python scripts/ingest.py --ticker AAPL --years 2022,2023,2024
    python scripts/ingest.py --ticker MSFT --form 10-Q --output-dir data/
"""

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.chunking.financial_chunker import FinancialChunker
from src.ingestion.edgar_client import EdgarClient
from src.ingestion.filing_parser import FilingParser
from src.ingestion.models import FilingMetadata
from src.retrieval.vector_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IDENTITY = "FinRAGAgent fin-rag-agent@example.com"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings and save parsed data as JSON."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol")
    parser.add_argument(
        "--form", default="10-K", help="SEC filing type (default: 10-K)"
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Comma-separated filing years (default: last 3 years)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/",
        help="Output directory for parsed filings (default: data/)",
    )
    return parser.parse_args()


def filter_filings_by_years(
    filings: list[FilingMetadata], years: list[int]
) -> list[FilingMetadata]:
    """Keep only filings whose filing_date year matches one of the target years."""
    return [f for f in filings if f.filing_date.year in years]


def ingest_ticker(
    ticker: str,
    form: str,
    years: list[int],
    output_dir: Path,
) -> None:
    """Download, parse, chunk, and index filings for a single ticker."""
    client = EdgarClient(identity=IDENTITY)
    parser = FilingParser()
    chunker = FinancialChunker()
    vector_store = ChromaStore()

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # Fetch enough filings to cover requested years
    logger.info("Fetching %s filings for %s...", form, ticker)
    all_filings = client.get_filings(
        ticker=ticker, filing_type=form, num_filings=10
    )
    logger.info("  Found %d %s filings total", len(all_filings), form)

    filings = filter_filings_by_years(all_filings, years)
    if not filings:
        logger.warning(
            "  No filings found for years %s. Available: %s",
            years,
            [f.filing_date.year for f in all_filings],
        )
        return

    logger.info(
        "  Processing %d filings for years %s",
        len(filings),
        sorted({f.filing_date.year for f in filings}),
    )

    for meta in filings:
        year = meta.filing_date.year
        logger.info(
            "  [%s %s %d] Downloading (accession: %s)...",
            ticker,
            form,
            year,
            meta.accession_number,
        )

        # Download markdown for section parsing
        try:
            markdown_text = client.download_filing_markdown(meta)
        except Exception:
            logger.exception("    Failed to download markdown, falling back to HTML")
            markdown_text = None

        # Download HTML for XBRL extraction
        try:
            html_text = client.download_filing(meta)
        except Exception:
            logger.exception("    Failed to download HTML")
            html_text = None

        # Parse sections from markdown (preferred) or HTML
        source_text = markdown_text or html_text or ""
        sections = parser.parse_sections(source_text, meta)

        # Extract XBRL facts from HTML
        xbrl_facts = []
        if html_text:
            xbrl_facts = parser.extract_xbrl_facts(html_text, meta)

        # Compute summary stats
        total_tables = sum(len(s.tables) for s in sections)
        unique_xbrl_concepts = {f.concept for f in xbrl_facts}

        logger.info(
            "    Sections: %d | Tables: %d | XBRL facts: %d (%d unique concepts)",
            len(sections),
            total_tables,
            len(xbrl_facts),
            len(unique_xbrl_concepts),
        )

        # Print section names
        for s in sections:
            table_note = f" ({len(s.tables)} tables)" if s.tables else ""
            logger.info("      - %s [%d chars]%s", s.section_name, len(s.content), table_note)

        # Save to JSON
        filing_data = {
            "metadata": meta.model_dump(mode="json"),
            "sections": [
                {
                    "section_name": s.section_name,
                    "content": s.content,
                    "tables": s.tables,
                }
                for s in sections
            ],
            "xbrl_facts": [
                {
                    "concept": f.concept,
                    "value": f.value,
                    "unit": f.unit,
                }
                for f in xbrl_facts
            ],
            "summary": {
                "num_sections": len(sections),
                "num_tables": total_tables,
                "num_xbrl_facts": len(xbrl_facts),
                "unique_xbrl_concepts": sorted(unique_xbrl_concepts),
            },
        }

        filename = f"{ticker}_{form}_{year}.json"
        out_path = ticker_dir / filename
        out_path.write_text(json.dumps(filing_data, indent=2, default=str), encoding="utf-8")
        logger.info("    Saved to %s", out_path)

        # Chunk and index into ChromaDB
        if sections:
            chunks = chunker.chunk_filing(sections)
            vector_store.add_chunks(chunks)
            logger.info("    Indexed %d chunks into ChromaDB", len(chunks))

        # Respect SEC rate limit (10 req/sec)
        time.sleep(0.5)


def main() -> None:
    """Run the ingestion pipeline from the command line."""
    args = parse_args()

    current_year = date.today().year
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    else:
        years = list(range(current_year - 2, current_year + 1))

    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("SEC EDGAR Ingestion Pipeline")
    logger.info("  Ticker: %s | Form: %s | Years: %s", args.ticker, args.form, years)
    logger.info("  Output: %s", output_dir.resolve())
    logger.info("=" * 60)

    start = time.time()
    ingest_ticker(args.ticker, args.form, years, output_dir)
    elapsed = time.time() - start

    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
