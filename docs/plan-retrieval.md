# Retrieval Pipeline — Implementation Plan

## 1. Existing State

### Stub modules (signatures defined, bodies are `...`):
- `src/retrieval/vector_store.py` — `VectorStore` (ChromaDB wrapper)
- `src/retrieval/bm25_search.py` — `BM25Search` (rank-bm25 wrapper)
- `src/retrieval/hybrid.py` — `HybridRetriever` (RRF fusion)
- `src/retrieval/reranker.py` — `CrossEncoderReranker`

### Existing models consumed by retrieval:
- `src/chunking/models.py` — `DocumentChunk`, `ChunkMetadata`
- `src/retrieval/models.py` — `SearchResult`, `RetrievalResult`
- `src/ingestion/models.py` — `XBRLFact`, `FilingMetadata`

### What the stubs are missing:
- Metadata filtering parameters on search methods
- XBRL fact retrieval (entirely absent)
- Query routing logic (numerical vs narrative)
- An orchestrator that wires all stages together
- Index management (update/delete individual chunks)

---

## 2. Model Changes

### 2.1 New model: `MetadataFilter` in `src/retrieval/models.py`

```python
class MetadataFilter(BaseModel):
    """Optional filters narrowing search to specific filings."""
    ticker: str | None = None
    year: int | None = None
    filing_type: str | None = None          # "10-K", "10-Q"
    section_name: str | None = None         # "Item 7", "Item 1A", etc.

    def to_chroma_where(self) -> dict | None:
        """Convert non-None fields to a ChromaDB $and where-clause dict."""

    def matches(self, meta: ChunkMetadata) -> bool:
        """In-memory match for BM25 pre/post-filtering."""
```

### 2.2 New enum: `QueryType` in `src/retrieval/models.py`

```python
class QueryType(str, Enum):
    NUMERICAL = "numerical"
    NARRATIVE = "narrative"
```

### 2.3 Extend `RetrievalResult`

Add field:
```python
    query_type: QueryType = QueryType.NARRATIVE
```

---

## 3. Module-by-Module Plan

### 3.1 `vector_store.py` — `VectorStore`

**Dependencies:** `chromadb`, `openai` (for embeddings via API key from `.env`)

#### `__init__`
- Load `OPENAI_API_KEY` from env via `python-dotenv`.
- Create `chromadb.PersistentClient(path=persist_dir)`.
- Get or create collection with name `collection_name`.
- Store `embedding_model` name for OpenAI calls.
- Use `openai.OpenAI().embeddings.create()` to produce embeddings. Wrap in a helper `_embed(texts: list[str]) -> list[list[float]]` that batches into groups of 100 (API limit per request).

#### `add_chunks(chunks: list[DocumentChunk]) -> None`
- Batch chunks in groups of 100 for ChromaDB upsert.
- For each batch call `_embed([c.content for c in batch])`.
- Call `collection.upsert(ids=..., embeddings=..., documents=..., metadatas=...)`.
- Metadata stored in ChromaDB: flatten `ChunkMetadata` fields to a flat dict (`ticker`, `year`, `filing_type`, `section_name`, `chunk_index`, `is_table`, `page_estimate`). ChromaDB requires flat string/int/float/bool values.

#### `search(query, top_k, metadata_filter: MetadataFilter | None) -> list[SearchResult]`
- Embed the query via `_embed([query])`.
- Build `where` clause from `metadata_filter.to_chroma_where()` if provided.
- Call `collection.query(query_embeddings=..., n_results=top_k, where=...)`.
- Map ChromaDB results to `SearchResult` objects. Set `source="vector"`. ChromaDB returns distances; convert to similarity score: `score = 1 - distance` (for cosine) or use `1 / (1 + distance)` for L2.

#### `delete_chunks(chunk_ids: list[str]) -> None`
- Call `collection.delete(ids=chunk_ids)`.

#### `delete_collection() -> None`
- Call `client.delete_collection(collection_name)`.

#### Error handling
- Wrap OpenAI embedding calls in try/except. On `openai.RateLimitError`, log warning and retry with exponential backoff (max 3 retries).
- On `chromadb` exceptions, log and re-raise as a custom `RetrievalError`.

---

### 3.2 `bm25_search.py` — `BM25Search`

**Dependencies:** `rank_bm25.BM25Okapi`

#### Internal state
```python
self._chunks: list[DocumentChunk]       # stored corpus for id/metadata lookup
self._bm25: BM25Okapi | None            # the index object
self._tokenized_corpus: list[list[str]]  # pre-tokenized docs
```

#### `build_index(chunks: list[DocumentChunk]) -> None`
- Store `chunks` reference.
- Tokenize each `chunk.content` with `_tokenize()` (lowercase, split on whitespace + punctuation, strip tokens < 2 chars).
- Create `BM25Okapi(tokenized_corpus)`.

#### `search(query, top_k, metadata_filter: MetadataFilter | None) -> list[SearchResult]`
- Tokenize query with same `_tokenize()`.
- Get scores: `self._bm25.get_scores(tokenized_query)` → numpy array of scores for every document.
- **Metadata filtering strategy — post-filter approach:**
  - Retrieve `top_k * 3` candidates by score.
  - Filter candidates where `metadata_filter.matches(chunk.metadata)` is True.
  - Truncate to `top_k`.
  - Rationale: BM25 scoring is global (needs all docs), so pre-filtering would require rebuilding the index. Post-filtering with over-fetch is simpler and sufficient given corpus size (~thousands of chunks per ticker).
- Map to `SearchResult` with `source="bm25"`.

#### `add_chunks(chunks: list[DocumentChunk]) -> None`
- Append to `self._chunks` and rebuild index. BM25Okapi does not support incremental add.

#### `remove_chunks(chunk_ids: set[str]) -> None`
- Filter `self._chunks` and rebuild index.

#### Error handling
- If `search()` is called before `build_index()`, raise `RetrievalError("BM25 index not built")`.

---

### 3.3 `hybrid.py` — `HybridRetriever`

**Dependencies:** `VectorStore`, `BM25Search`

#### `__init__`
```python
def __init__(
    self,
    vector_store: VectorStore,
    bm25_search: BM25Search,
    rrf_k: int = 60,
) -> None:
```
Accept pre-built `VectorStore` and `BM25Search` via composition.

#### `fuse(vector_results, bm25_results, top_k) -> list[SearchResult]`

**RRF algorithm:**
```
For each result list L, assign rank r (1-based) to each document.
RRF_score(doc) = Σ  1 / (k + r_L(doc))   for each list L containing doc
```

Implementation:
1. Build `scores: dict[str, float]` keyed by `chunk_id`.
2. For each `(rank, result)` in `enumerate(vector_results, start=1)`:
   `scores[result.chunk_id] += 1.0 / (self.rrf_k + rank)`
3. Same for `bm25_results`.
4. Also maintain `docs: dict[str, SearchResult]` to hold the full result objects. When a chunk_id appears in both lists, keep whichever has the higher original score.
5. Sort by RRF score descending, take `top_k`.
6. Replace each result's `score` with its RRF score. Set `source="rrf"`.

#### `retrieve(query, top_k, metadata_filter) -> RetrievalResult`
1. Call `vector_store.search(query, top_k=top_k * 2, metadata_filter=metadata_filter)`.
2. Call `bm25_search.search(query, top_k=top_k * 2, metadata_filter=metadata_filter)`.
3. Call `self.fuse(vector_results, bm25_results, top_k)`.
4. Return `RetrievalResult(query=query, results=fused, stages_applied=["vector", "bm25", "rrf"])`.

Over-fetch factor of 2× ensures RRF has enough candidates from each source.

---

### 3.4 `reranker.py` — `CrossEncoderReranker`

**Dependencies:** `sentence_transformers.CrossEncoder`

#### `__init__`
- Load model: `CrossEncoder(model_name)`.
- Log model load time (these models are ~80MB, one-time cost).

#### `rerank(query, results, top_k) -> list[SearchResult]`
1. Build pairs: `[(query, r.text) for r in results]`.
2. Predict: `scores = self.model.predict(pairs)` → array of float scores.
3. Assign `results[i].score = float(scores[i])` and `results[i].source = "rerank"`.
4. Sort descending by score, return `top_k`.

#### Error handling
- If `results` is empty, return `[]` immediately.
- Wrap `model.predict` in try/except. On failure, log error and return the input list unchanged (graceful degradation — better to return un-reranked results than nothing).

---

### 3.5 New module: `src/retrieval/xbrl_store.py` — `XBRLFactStore`

Handles structured XBRL data for precise numerical lookups. This is a separate retrieval path from the chunk-based similarity search.

```python
class XBRLFactStore:
    """In-memory store for XBRL facts with exact-match lookups."""

    def __init__(self) -> None:
        self._facts: list[XBRLFact]          # all loaded facts
        self._concept_index: dict[str, list[int]]  # concept → fact indices

    def add_facts(self, facts: list[XBRLFact]) -> None:
        """Add XBRL facts and rebuild concept index."""

    def search(
        self,
        concept: str | None = None,
        ticker: str | None = None,
        year: int | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Exact-match lookup of XBRL facts.

        Filters by concept (substring match, case-insensitive),
        ticker, and year. Returns matching facts formatted as
        SearchResult objects with source='xbrl'.

        Score: 1.0 for exact concept match, 0.8 for substring match.
        Text: formatted string like
            'Revenue (us-gaap:Revenue): $394,328M USD
             Period: 2024-01-01 to 2024-12-31
             Filing: AAPL 10-K 2024'
        """

    def remove_facts(
        self,
        ticker: str | None = None,
        year: int | None = None,
    ) -> int:
        """Remove facts matching filters. Returns count removed."""
```

**Why not ChromaDB for XBRL?** Numerical facts need exact-match lookups by concept name, not semantic similarity. A user asking "What was Apple's revenue in 2024?" needs the specific `us-gaap:Revenue` fact, not the 5 chunks most semantically similar to "revenue". The in-memory index is small (hundreds of facts per filing) and fast.

---

### 3.6 New module: `src/retrieval/query_router.py` — `QueryRouter`

Routes queries to the appropriate retrieval path based on query characteristics.

```python
class QueryRouter:
    """Classifies queries and routes to appropriate retrieval path."""

    # Keyword signals for numerical queries
    NUMERICAL_SIGNALS: list[str]  # e.g., "revenue", "earnings", "EPS",
                                  # "net income", "how much", "what was the",
                                  # "total assets", "margin", "EBITDA", "debt",
                                  # "cash flow", "shares outstanding", "$", "%"

    XBRL_CONCEPT_MAP: dict[str, str]
    # Maps common financial terms to XBRL concepts:
    #   "revenue" → "us-gaap:Revenue"
    #   "net income" → "us-gaap:NetIncomeLoss"
    #   "total assets" → "us-gaap:Assets"
    #   "eps" → "us-gaap:EarningsPerShareBasic"
    #   etc. (~20-30 common mappings)

    def classify(self, query: str) -> QueryType:
        """Classify a query as numerical or narrative.

        Heuristic approach (no LLM call needed):
        1. Check for presence of NUMERICAL_SIGNALS (case-insensitive).
        2. Check for number patterns (digits, $, %).
        3. Check for comparison language ("compared to", "year over year").
        4. If ≥2 signals match → NUMERICAL.
        5. Default → NARRATIVE.
        """

    def extract_filters(self, query: str) -> MetadataFilter:
        """Extract metadata filters from natural language query.

        Regex-based extraction:
        - Ticker: uppercase 1-5 letter words matching known tickers,
          or words following "for"/"of"
        - Year: 4-digit numbers 2000-2030
        - Filing type: "10-K", "10-Q", "8-K" patterns
        - Section: "risk factors" → "Item 1A", "MD&A" → "Item 7", etc.
        """

    def map_to_xbrl_concept(self, query: str) -> str | None:
        """Attempt to map query terms to an XBRL concept name.

        Returns the XBRL concept string if a match is found, else None.
        Uses XBRL_CONCEPT_MAP with fuzzy keyword matching.
        """
```

**Why heuristic, not LLM-based?** Latency. Every retrieval call would add an LLM round-trip (~500ms+). The keyword heuristic covers >90% of financial query patterns. If heuristic accuracy proves insufficient later, we can swap in an LLM classifier behind the same interface.

---

### 3.7 New module: `src/retrieval/pipeline.py` — `RetrievalPipeline`

Top-level orchestrator that wires everything together. This is the single entry point the agent layer calls.

```python
class RetrievalPipeline:
    """End-to-end retrieval pipeline with query routing.

    Orchestrates: query classification → metadata extraction →
    retrieval (XBRL and/or hybrid) → reranking → final results.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_search: BM25Search,
        xbrl_store: XBRLFactStore,
        reranker: CrossEncoderReranker,
        rrf_k: int = 60,
        hybrid_top_k: int = 20,    # candidates passed to reranker
        final_top_k: int = 5,      # results returned to caller
    ) -> None:

    def query(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Execute the full retrieval pipeline.

        Flow:
        1. QueryRouter.classify(query) → QueryType
        2. QueryRouter.extract_filters(query) → auto_filter
           Merge auto_filter with explicit metadata_filter (explicit wins).
        3. Route:
           a. NUMERICAL path:
              - xbrl_store.search(concept, ticker, year)
              - If XBRL returns ≥1 result, prepend to results.
              - ALSO run hybrid search (user may want context around the number).
           b. NARRATIVE path:
              - hybrid search only.
        4. HybridRetriever.retrieve(query, hybrid_top_k, merged_filter)
        5. CrossEncoderReranker.rerank(query, hybrid_results, final_top_k)
        6. For NUMERICAL: merge XBRL results (always at top) + reranked hybrid.
        7. Return RetrievalResult with all stages logged.
        """

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add chunks to both vector store and BM25 index."""

    def index_xbrl_facts(self, facts: list[XBRLFact]) -> None:
        """Add XBRL facts to the fact store."""

    def remove_filing(
        self,
        ticker: str,
        year: int | None = None,
    ) -> None:
        """Remove all data for a filing from all indices.

        1. Query ChromaDB for chunk_ids matching ticker+year.
        2. Delete those chunk_ids from vector store.
        3. Remove matching chunks from BM25 and rebuild.
        4. Remove matching facts from XBRL store.
        """
```

---

## 4. Data Flow Diagram

```
User Query
    │
    ▼
┌─────────────────┐
│  QueryRouter     │──→ QueryType (NUMERICAL | NARRATIVE)
│  .classify()     │──→ MetadataFilter (auto-extracted)
│  .extract_filters│──→ XBRL concept (if numerical)
└────────┬────────┘
         │
         ▼
    ┌────────────── NUMERICAL? ──────────────┐
    │ YES                                     │ NO
    ▼                                         │
┌──────────────┐                              │
│ XBRLFactStore│                              │
│ .search()    │──→ xbrl_results              │
└──────┬───────┘                              │
       │          (also run hybrid)           │
       ▼                                      ▼
┌──────────────────────────────────────────────┐
│              HybridRetriever.retrieve()      │
│  ┌─────────────┐     ┌─────────────────┐     │
│  │ VectorStore │     │   BM25Search    │     │
│  │ .search()   │     │   .search()     │     │
│  │ (ChromaDB   │     │   (rank-bm25    │     │
│  │  +OpenAI    │     │    in-memory)   │     │
│  │  embeddings)│     │                 │     │
│  └──────┬──────┘     └───────┬─────────┘     │
│         │                    │               │
│         └────────┬───────────┘               │
│                  ▼                           │
│         ┌────────────────┐                   │
│         │  RRF Fusion    │                   │
│         │  k=60          │                   │
│         │  1/(k+rank)    │                   │
│         └───────┬────────┘                   │
└─────────────────┼────────────────────────────┘
                  ▼
         ┌────────────────────┐
         │ CrossEncoderReranker│
         │ ms-marco-MiniLM    │
         │ .rerank()           │
         └────────┬───────────┘
                  ▼
         ┌────────────────┐
         │ Merge:         │
         │ XBRL (top)     │
         │ + reranked     │
         └────────┬───────┘
                  ▼
           RetrievalResult
```

---

## 5. RRF Implementation Detail

**Formula:** `RRF_score(d) = Σ_L 1 / (k + rank_L(d))`

With `k=60` and two lists (vector, BM25):
- Doc ranked #1 in both lists: `1/61 + 1/61 = 0.0328`
- Doc ranked #1 in vector only: `1/61 = 0.0164`
- Doc ranked #10 in vector, #3 in BM25: `1/70 + 1/63 = 0.0302`

This means a doc appearing in both lists at moderate ranks beats a doc appearing at #1 in only one list — which is the desired behavior for hybrid fusion.

**Implementation notes:**
- Use `defaultdict(float)` for accumulation.
- Rank is 1-based (first result = rank 1).
- Ties in RRF score: break by keeping the order of the higher-ranked list (vector first, as it tends to have better semantic matches).

---

## 6. Metadata Filtering Integration

### ChromaDB (dense search)
ChromaDB supports `where` clauses natively. `MetadataFilter.to_chroma_where()` produces:
```python
# Example: ticker="AAPL", year=2024
{"$and": [{"ticker": {"$eq": "AAPL"}}, {"year": {"$eq": 2024}}]}
# Single field: {"ticker": {"$eq": "AAPL"}}
# No filters: None (omit where clause)
```
Filtering happens server-side inside ChromaDB before distance computation — efficient.

### BM25 (sparse search)
rank-bm25 has no built-in filtering. Two viable approaches:

**Chosen: Post-filter with over-fetch**
- Retrieve `top_k * 3` results from BM25.
- Apply `metadata_filter.matches()` to each.
- Return the first `top_k` that pass.
- If fewer than `top_k` pass, return what we have.

**Rejected alternative: Pre-filter and rebuild index.**
- Would require building a subset BM25 index per query. Too slow for interactive use (~100ms to rebuild vs ~1ms to post-filter).

### XBRL (fact store)
Direct field matching. Already filtered by ticker/year/concept in the `search()` method.

---

## 7. Numerical vs Narrative Query Detection

### Heuristic signals (checked case-insensitively):

| Category | Signals |
|----------|---------|
| Financial metrics | revenue, earnings, EPS, net income, gross profit, operating income, EBITDA, cash flow, total assets, total liabilities, debt, margin, ratio |
| Question patterns | "how much", "what was the", "what is the", "how many" |
| Symbols | `$`, `%`, digits in context of amounts |
| Comparisons | "compared to", "year over year", "YoY", "growth", "increase", "decrease", "change in" |

### Classification rule:
```
signal_count = count of distinct signals found in query
if signal_count >= 2: return NUMERICAL
if signal_count == 1 and query contains a year: return NUMERICAL
else: return NARRATIVE
```

### Both paths always run hybrid search
Even for NUMERICAL queries, we run hybrid search to provide contextual passages around the numbers. XBRL results are pinned to the top of the final result list (they are the authoritative answer), followed by relevant narrative chunks that provide context.

---

## 8. Error Handling Strategy

### Custom exception hierarchy in `src/retrieval/errors.py`:

```python
class RetrievalError(Exception):
    """Base exception for retrieval failures."""

class EmbeddingError(RetrievalError):
    """OpenAI embedding API failure."""

class IndexNotBuiltError(RetrievalError):
    """BM25 search called before index is built."""

class VectorStoreError(RetrievalError):
    """ChromaDB operation failure."""
```

### Per-component error policy:

| Component | On failure | Rationale |
|-----------|-----------|-----------|
| OpenAI embeddings | Retry 3× with backoff, then raise `EmbeddingError` | Transient API errors are common |
| ChromaDB query | Raise `VectorStoreError` | Persistent store corruption needs attention |
| BM25 search | Raise `IndexNotBuiltError` if no index | Programmer error, fail fast |
| Cross-encoder | Log error, return un-reranked results | Reranking is an optimization, not critical path |
| XBRL store | Return empty list | Missing facts shouldn't block narrative search |
| Query router | Default to NARRATIVE | Safe fallback — hybrid search works for all queries |
| Pipeline orchestrator | If vector fails, fall back to BM25-only (and vice versa). If both fail, raise. | At least one search path should work |

### Logging
- Use `logging.getLogger(__name__)` in each module.
- Log all OpenAI API calls with token counts (per CLAUDE.md: "Log all LLM calls with input/output for debugging").
- Log query classification decisions at DEBUG level.
- Log retrieval timing per stage at INFO level.

---

## 9. Index Management

### Adding documents
```
ingest.py → FilingSections + XBRLFacts
         → chunker → list[DocumentChunk]
         → pipeline.index_chunks(chunks)       # → VectorStore + BM25
         → pipeline.index_xbrl_facts(facts)    # → XBRLFactStore
```

### Updating a filing (re-ingest)
1. `pipeline.remove_filing(ticker="AAPL", year=2024)` — clears old data from all three stores.
2. Re-run ingestion and indexing as above.
3. No partial updates — always replace the full filing.

### BM25 index persistence
BM25 index is in-memory only. On startup:
1. Load all `DocumentChunk` records from ChromaDB (using `collection.get()`).
2. Rebuild BM25 index from loaded chunks.
3. This is a known cold-start cost (~1-2s for 10K chunks). Acceptable for this project scope.

### ChromaDB persistence
ChromaDB persists to disk automatically via `PersistentClient`. The `.chroma/` directory is the source of truth for chunk data.

---

## 10. File Changes Summary

| File | Action | What changes |
|------|--------|-------------|
| `src/retrieval/models.py` | Edit | Add `MetadataFilter`, `QueryType` enum, extend `RetrievalResult` |
| `src/retrieval/vector_store.py` | Edit | Implement `VectorStore`, add `metadata_filter` param, add `delete_chunks` |
| `src/retrieval/bm25_search.py` | Edit | Implement `BM25Search`, add `metadata_filter` param, add `add_chunks`/`remove_chunks` |
| `src/retrieval/hybrid.py` | Edit | Implement RRF fusion, accept `VectorStore`+`BM25Search` in `__init__`, add `metadata_filter` param |
| `src/retrieval/reranker.py` | Edit | Implement cross-encoder reranking |
| `src/retrieval/xbrl_store.py` | Create | `XBRLFactStore` for numerical fact lookups |
| `src/retrieval/query_router.py` | Create | `QueryRouter` with heuristic classification |
| `src/retrieval/pipeline.py` | Create | `RetrievalPipeline` orchestrator |
| `src/retrieval/errors.py` | Create | Custom exception hierarchy |
| `src/retrieval/__init__.py` | Edit | Export public classes |

---

## 11. Dependencies to Add

```
chromadb>=0.4.0
openai>=1.0.0
rank-bm25>=0.2.2
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
```

---

## 12. Implementation Order

1. **`errors.py`** — exceptions first, everything depends on them.
2. **`models.py`** — add `MetadataFilter` and `QueryType`.
3. **`vector_store.py`** — ChromaDB + OpenAI embeddings. Test: add chunks, search, filter, delete.
4. **`bm25_search.py`** — BM25 index. Test: build, search, filter, rebuild.
5. **`hybrid.py`** — RRF fusion. Test: fuse mock results, verify score calculation.
6. **`reranker.py`** — Cross-encoder. Test: rerank mock results, verify ordering.
7. **`xbrl_store.py`** — Fact store. Test: add facts, lookup by concept/ticker/year.
8. **`query_router.py`** — Classification. Test: numerical vs narrative queries, filter extraction.
9. **`pipeline.py`** — Wire everything. Integration test: end-to-end query flow.
10. **`__init__.py`** — Exports.
