"""Tests for the retrieval package (vector store, BM25, hybrid, reranker)."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from src.chunking.models import ChunkMetadata, DocumentChunk
from src.retrieval.bm25_search import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.models import RetrievalConfig, SearchResult
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import ChromaStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str,
    ticker: str = "AAPL",
    year: int = 2024,
    chunk_id: str | None = None,
    section_name: str = "Item 7",
    chunk_index: int = 0,
    is_table: bool = False,
) -> DocumentChunk:
    """Create a DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id=chunk_id or hashlib.md5(content.encode()).hexdigest(),
        content=content,
        metadata=ChunkMetadata(
            ticker=ticker,
            year=year,
            filing_type="10-K",
            section_name=section_name,
            chunk_index=chunk_index,
            is_table=is_table,
        ),
    )


def _make_result(
    chunk_id: str,
    content: str,
    score: float,
    source: str = "dense",
    is_table: bool = False,
) -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata=ChunkMetadata(
            ticker="AAPL",
            year=2024,
            filing_type="10-K",
            section_name="Item 7",
            chunk_index=0,
            is_table=is_table,
        ),
        source=source,
    )


def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic mock embeddings based on text hash."""
    results = []
    for text in texts:
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vec = [float(b) / 255.0 for b in hash_bytes]
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        results.append(vec)
    return results


@pytest.fixture
def store(tmp_path):
    """Create a ChromaStore with mocked embedding model."""
    with patch("src.retrieval.vector_store.OpenAIEmbedding"):
        s = ChromaStore(
            persist_dir=str(tmp_path / ".chroma"),
            collection_name="test_collection",
        )
    s._embed = _mock_embed
    return s


# ---------------------------------------------------------------------------
# ChromaStore tests
# ---------------------------------------------------------------------------


class TestChromaStore:
    """Tests for ChromaStore."""

    def test_add_and_search(self, store: ChromaStore) -> None:
        """Test that indexed chunks are retrievable via search."""
        chunks = [
            _make_chunk(
                "Apple reported record revenue of $394 billion", chunk_id="c1"
            ),
            _make_chunk(
                "Risk factors include supply chain disruptions", chunk_id="c2"
            ),
            _make_chunk(
                "Management discussion of financial operations", chunk_id="c3"
            ),
        ]
        store.add_chunks(chunks)

        results = store.search("revenue earnings financial results")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == "dense" for r in results)
        for r in results:
            assert r.chunk_id in {"c1", "c2", "c3"}
            assert len(r.content) > 0
            assert r.metadata.ticker == "AAPL"
            assert r.metadata.year == 2024

    def test_metadata_filtering(self, store: ChromaStore) -> None:
        """Test that search respects metadata filters."""
        chunks = [
            _make_chunk(
                "Apple revenue data for fiscal year", ticker="AAPL", chunk_id="c1"
            ),
            _make_chunk(
                "Microsoft revenue data for fiscal year",
                ticker="MSFT",
                chunk_id="c2",
            ),
            _make_chunk(
                "Google revenue data for fiscal year",
                ticker="GOOGL",
                chunk_id="c3",
            ),
        ]
        store.add_chunks(chunks)

        results = store.search("revenue", filters={"ticker": "AAPL"})
        assert len(results) == 1
        assert results[0].metadata.ticker == "AAPL"
        assert results[0].chunk_id == "c1"

    def test_delete_by_ticker(self, store: ChromaStore) -> None:
        """Test that delete_by_ticker removes only matching chunks."""
        chunks = [
            _make_chunk("Apple data chunk one", ticker="AAPL", chunk_id="c1"),
            _make_chunk("Apple data chunk two", ticker="AAPL", chunk_id="c2"),
            _make_chunk("Microsoft data chunk", ticker="MSFT", chunk_id="c3"),
        ]
        store.add_chunks(chunks)

        stats = store.get_collection_stats()
        assert stats["count"] == 3
        assert set(stats["tickers"]) == {"AAPL", "MSFT"}

        store.delete_by_ticker("AAPL")

        stats = store.get_collection_stats()
        assert stats["count"] == 1
        assert stats["tickers"] == ["MSFT"]
        assert "AAPL" not in stats["tickers"]


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------


class TestBM25Tokenizer:
    """Tests for improved BM25Index._tokenize."""

    def test_sec_filing_type_preserved(self) -> None:
        """Test '10-K' stays as single token."""
        tokens = BM25Index._tokenize("10-K filing")
        assert "10-k" in tokens

    def test_dollar_amount(self) -> None:
        """Test '$391.0 billion' preserves the number without $."""
        tokens = BM25Index._tokenize("$391.0 billion")
        assert "391.0" in tokens
        assert "billion" in tokens

    def test_percentage(self) -> None:
        """Test '46.2%' becomes '46.2' without %."""
        tokens = BM25Index._tokenize("46.2% growth")
        assert "46.2" in tokens
        assert "growth" in tokens

    def test_hyphenated_compound(self) -> None:
        """Test 'year-over-year' stays as single token."""
        tokens = BM25Index._tokenize("year-over-year growth")
        assert "year-over-year" in tokens
        assert "growth" in tokens

    def test_long_term_preserved(self) -> None:
        """Test 'long-term' stays as single token."""
        tokens = BM25Index._tokenize("long-term debt")
        assert "long-term" in tokens


class TestBM25Persistence:
    """Tests for BM25Index save/load persistence."""

    def _build_sample_index(self) -> BM25Index:
        """Build a small BM25 index for persistence tests."""
        chunks = [
            _make_chunk("Apple reported record revenue of $394 billion", chunk_id="c1"),
            _make_chunk("Risk factors include supply chain disruptions", chunk_id="c2"),
            _make_chunk("Revenue and net income reached record levels", chunk_id="c3"),
        ]
        index = BM25Index()
        index.build_index(chunks)
        return index

    def test_save_creates_file(self, tmp_path) -> None:
        """Test save_index creates a file at the expected path."""
        index = self._build_sample_index()
        path = tmp_path / "bm25_index.pkl"
        index.save_index(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_restores_functional_index(self, tmp_path) -> None:
        """Test load_index restores a functional BM25Index."""
        index = self._build_sample_index()
        path = tmp_path / "bm25_index.pkl"
        index.save_index(path)

        loaded = BM25Index.load_index(path)
        results = loaded.search("revenue", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_load_raises_file_not_found(self, tmp_path) -> None:
        """Test load_index raises FileNotFoundError for missing path."""
        missing_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            BM25Index.load_index(missing_path)

    def test_roundtrip_produces_identical_results(self, tmp_path) -> None:
        """Test build -> save -> load -> search produces identical results."""
        index = self._build_sample_index()
        original_results = index.search("revenue growth", top_k=3)

        path = tmp_path / "bm25_index.pkl"
        index.save_index(path)
        loaded = BM25Index.load_index(path)
        loaded_results = loaded.search("revenue growth", top_k=3)

        assert len(loaded_results) == len(original_results)
        for orig, restored in zip(original_results, loaded_results):
            assert orig.chunk_id == restored.chunk_id
            assert orig.content == restored.content
            assert orig.score == pytest.approx(restored.score)
            assert orig.metadata.ticker == restored.metadata.ticker

    def test_save_creates_parent_directories(self, tmp_path) -> None:
        """Test save creates parent directories if they don't exist."""
        index = self._build_sample_index()
        path = tmp_path / "nested" / "dirs" / "bm25_index.pkl"
        index.save_index(path)
        assert path.exists()


class TestBM25Index:
    """Tests for BM25Index."""

    def test_bm25_search(self) -> None:
        """Test that BM25 index returns keyword-relevant results."""
        chunks = [
            _make_chunk(
                "Apple revenue grew significantly in fiscal 2024", chunk_id="c1"
            ),
            _make_chunk(
                "Risk factors include global supply chain disruptions", chunk_id="c2"
            ),
            _make_chunk(
                "Revenue and net income reached record levels", chunk_id="c3"
            ),
        ]
        index = BM25Index()
        index.build_index(chunks)

        results = index.search("revenue growth", top_k=3)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == "sparse" for r in results)
        # "revenue" appears in c1 and c3 — both should be returned
        chunk_ids = [r.chunk_id for r in results]
        assert "c1" in chunk_ids
        assert "c3" in chunk_ids


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_hybrid_fusion_ordering(self) -> None:
        """Test that RRF fusion ranks docs appearing in both lists highest."""
        dense_results = [
            _make_result("c1", "doc1", score=0.9, source="dense"),
            _make_result("c2", "doc2", score=0.8, source="dense"),
            _make_result("c3", "doc3", score=0.7, source="dense"),
        ]
        sparse_results = [
            _make_result("c2", "doc2", score=5.0, source="sparse"),
            _make_result("c4", "doc4", score=4.0, source="sparse"),
            _make_result("c1", "doc1", score=3.0, source="sparse"),
        ]

        mock_vs = MagicMock()
        mock_vs.search.return_value = dense_results
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = sparse_results

        retriever = HybridRetriever(mock_vs, mock_bm25)
        results = retriever.search("test query")

        assert len(results) == 4
        assert all(r.source == "hybrid" for r in results)
        # c1 and c2 appear in both lists → highest RRF scores
        top_ids = [r.chunk_id for r in results[:2]]
        assert set(top_ids) == {"c1", "c2"}
        # Scores should decrease
        assert results[0].score >= results[1].score >= results[2].score

    def test_expand_query_with_revenue(self) -> None:
        """Test query with 'revenue' expands to include synonyms."""
        expanded = HybridRetriever._expand_query("What was Apple's revenue?")
        assert "net sales" in expanded
        assert "total revenue" in expanded
        assert "net revenue" in expanded
        # Original query preserved
        assert expanded.startswith("What was Apple's revenue?")

    def test_expand_query_no_match(self) -> None:
        """Test query with no matching terms returns unchanged."""
        query = "Tell me about Apple's management team"
        assert HybridRetriever._expand_query(query) == query

    def test_expand_query_multiple_terms(self) -> None:
        """Test query with multiple matching terms expands all."""
        expanded = HybridRetriever._expand_query("Compare revenue and debt")
        assert "net sales" in expanded
        assert "liabilities" in expanded

    def test_reranker_is_called_when_provided(self) -> None:
        """Test that HybridRetriever delegates to reranker when one is set."""
        dense_results = [
            _make_result("c1", "doc1", score=0.9, source="dense"),
            _make_result("c2", "doc2", score=0.8, source="dense"),
        ]
        sparse_results = [
            _make_result("c2", "doc2", score=5.0, source="sparse"),
        ]

        mock_vs = MagicMock()
        mock_vs.search.return_value = dense_results
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = sparse_results

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            _make_result("c2", "doc2", score=1.0, source="rerank"),
        ]

        retriever = HybridRetriever(mock_vs, mock_bm25, reranker=mock_reranker)
        results = retriever.search("test query")

        mock_reranker.rerank.assert_called_once()
        assert results[0].chunk_id == "c2"

    def test_no_reranker_returns_top_k(self) -> None:
        """Test that HybridRetriever without reranker returns fused[:top_k]."""
        dense_results = [
            _make_result("c1", "doc1", score=0.9, source="dense"),
            _make_result("c2", "doc2", score=0.8, source="dense"),
        ]
        sparse_results = []

        mock_vs = MagicMock()
        mock_vs.search.return_value = dense_results
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = sparse_results

        config = RetrievalConfig(top_k=1)
        retriever = HybridRetriever(mock_vs, mock_bm25, config, reranker=None)
        results = retriever.search("test query")

        assert len(results) == 1

    def test_detect_target_sections_balance_sheet(self) -> None:
        """Test 'balance sheet' routes to Item 8."""
        sections = HybridRetriever._detect_target_sections("What is on the balance sheet?")
        assert "Item 8" in sections

    def test_detect_target_sections_risk(self) -> None:
        """Test 'risk factors' routes to Item 1A."""
        sections = HybridRetriever._detect_target_sections("What are the risk factors?")
        assert "Item 1A" in sections

    def test_detect_target_sections_revenue(self) -> None:
        """Test 'revenue' routes to Item 7 and Item 8."""
        sections = HybridRetriever._detect_target_sections("What was the revenue?")
        assert "Item 7" in sections
        assert "Item 8" in sections

    def test_detect_target_sections_no_match(self) -> None:
        """Test generic query returns empty list."""
        sections = HybridRetriever._detect_target_sections("Tell me about the company")
        assert sections == []

    def test_section_boost_increases_matching_score(self) -> None:
        """Test that results with matching sections get boosted above non-matching."""
        item8_result = SearchResult(
            chunk_id="c1",
            content="Balance sheet data",
            score=0.5,
            metadata=ChunkMetadata(
                ticker="AAPL", year=2024, filing_type="10-K",
                section_name="Item 8. Financial Statements", chunk_index=0,
            ),
            source="rerank",
        )
        item7_result = SearchResult(
            chunk_id="c2",
            content="MD&A discussion",
            score=0.6,
            metadata=ChunkMetadata(
                ticker="AAPL", year=2024, filing_type="10-K",
                section_name="Item 7. MD&A", chunk_index=0,
            ),
            source="rerank",
        )

        mock_vs = MagicMock()
        mock_vs.search.return_value = [item8_result, item7_result]
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = []

        retriever = HybridRetriever(mock_vs, mock_bm25)
        # "balance sheet" → boosts Item 8 before reranking
        results = retriever.search("balance sheet assets")

        # Item 8 chunk should now rank first due to section boost
        # (RRF score boosted by 1.3x moves it above Item 7)
        assert results[0].chunk_id == "c1"
        assert results[0].score > results[1].score

    def test_numerical_query_routing(self) -> None:
        """Test that numerical queries boost table chunk scores above text."""
        dense_results = [
            _make_result(
                "c1", "narrative about revenue", score=0.9, source="dense"
            ),
            _make_result(
                "c2",
                "| Revenue | $394B |",
                score=0.7,
                source="dense",
                is_table=True,
            ),
        ]
        sparse_results = [
            _make_result(
                "c1", "narrative about revenue", score=5.0, source="sparse"
            ),
        ]

        mock_vs = MagicMock()
        mock_vs.search.return_value = dense_results
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = sparse_results

        retriever = HybridRetriever(mock_vs, mock_bm25)
        # "how much" (strong=2) + "revenue" (weak=1) → score 3 → boost
        results = retriever.search("How much revenue did they earn in 2024")

        assert len(results) == 2
        # Table chunk (c2) should be boosted to rank 1
        assert results[0].metadata.is_table
        assert results[0].chunk_id == "c2"


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


class TestReranker:
    """Tests for Reranker."""

    def test_reranking_improves_relevance(self) -> None:
        """Test that reranker reorders by cross-encoder score and normalizes."""
        results = [
            _make_result("c1", "irrelevant noise", score=0.9, source="hybrid"),
            _make_result("c2", "Apple revenue analysis", score=0.8, source="hybrid"),
            _make_result(
                "c3", "Apple Q4 revenue was $394B", score=0.7, source="hybrid"
            ),
        ]

        with patch("src.retrieval.reranker.CrossEncoder") as MockCE:
            # Mock: c3 scored highest, c2 middle, c1 lowest
            MockCE.return_value.predict.return_value = [0.1, 0.6, 0.9]
            reranker = Reranker()
            reranked = reranker.rerank("Apple revenue", results, top_k=2)

        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c3"
        assert reranked[1].chunk_id == "c2"
        assert all(r.source == "rerank" for r in reranked)
        # Sigmoid-normalized: c3 → sigmoid(0.9) ≈ 0.7109, c2 → sigmoid(0.6) ≈ 0.6457
        assert reranked[0].score == pytest.approx(0.7109, abs=1e-3)
        assert reranked[1].score == pytest.approx(0.6457, abs=1e-3)
