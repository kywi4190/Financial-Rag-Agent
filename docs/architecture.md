# Architecture & Design

Comprehensive technical documentation for the Financial Document RAG Agent.

---

## System Architecture

### Data Flow

```
User Query
    │
    ▼
┌────────────────────────────────┐
│  FinancialQueryEngine (CRAG)   │
│  1. Retrieve context           │
│  2. Score relevance (LLM)      │
│  3. If < 0.6 → reformulate    │
│     and re-retrieve            │
│  4. Generate answer + citations│
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Hybrid Retrieval Pipeline     │
│                                │
│  Dense (ChromaDB)  BM25 Sparse │
│       │               │       │
│       └──► RRF Fusion ◄┘      │
│               │                │
│               ▼                │
│     Cross-Encoder Reranker     │
│     (ms-marco-MiniLM-L-6-v2)  │
└────────────┬───────────────────┘
             │
             ▼
┌────────────────────────────────┐
│  Storage Layer                 │
│  ChromaDB  │  BM25   │  XBRL  │
│  (vectors) │ (terms) │  (DFs) │
└────────────┴─────────┴────────┘
             ▲
             │
┌────────────┴───────────────────┐
│  Ingestion Pipeline            │
│  SEC EDGAR → Parser → Chunker  │
│  → Embedder → Index            │
└────────────────────────────────┘
```

### Component Responsibilities

| Component | Module | Role |
|---|---|---|
| EDGAR Client | `src/ingestion/edgar_client.py` | Downloads 10-K/10-Q filings from SEC EDGAR via edgartools |
| Filing Parser | `src/ingestion/filing_parser.py` | Extracts Item sections + inline XBRL facts from HTML/markdown |
| Financial Chunker | `src/chunking/financial_chunker.py` | Structure-aware splitting with atomic table chunks |
| Table Handler | `src/chunking/table_handler.py` | HTML table to markdown conversion + financial table classification |
| ChromaDB Store | `src/retrieval/vector_store.py` | Dense vector indexing with OpenAI embeddings, batched upsert |
| BM25 Index | `src/retrieval/bm25_search.py` | In-memory sparse keyword search using Okapi BM25 |
| Hybrid Retriever | `src/retrieval/hybrid.py` | RRF fusion of dense + sparse results with numerical query boosting |
| Reranker | `src/retrieval/reranker.py` | Cross-encoder re-scoring with min-max normalization |
| Query Engine | `src/agents/query_engine.py` | CRAG loop: retrieve → evaluate → reformulate → generate |
| Financial Tools | `src/agents/financial_tools.py` | XBRL lookup, ratio calculation, metric comparison tools |
| Memo Generator | `src/agents/memo_generator.py` | 3-agent orchestration for investment memo production |
| RAGAS Evaluator | `src/evaluation/ragas_eval.py` | Standard + custom financial metrics evaluation |
| Streamlit App | `app/main.py` | UI with chat, memo, and metrics tabs |

---

## Key Design Decisions

### 1. Hybrid Search (Dense + BM25 + RRF)

**Problem:** Financial documents contain exact terms (ticker symbols, XBRL concepts like `us-gaap:Revenues`, dollar amounts) that dense embeddings map to a generic "financial" neighborhood, losing lexical precision.

**Solution:** Dual retrieval with Reciprocal Rank Fusion.

- Dense search (ChromaDB, cosine HNSW) captures semantic similarity
- BM25 (Okapi) captures exact keyword matches
- RRF with k=60 fuses ranked lists: `score = sum(weight / (k + rank))`
- Default weights: dense=0.7, sparse=0.3
- Numerical query detection (2+ pattern matches) triggers 1.5x table chunk boost

**Tradeoff:** BM25 index lives in memory and must be rebuilt on cold start. Adds ~2s for 10K chunks. Acceptable for the current scale; persistence would be needed above ~100K chunks.

### 2. Structure-Aware Chunking

**Problem:** Fixed-size chunking (e.g., LangChain's RecursiveCharacterTextSplitter) splits mid-table, mid-paragraph, and across section boundaries, destroying the structural signals that make financial filings navigable.

**Solution:** Custom `FinancialChunker` that:

1. Splits first by filing section (Item 1, 1A, 7, 8, etc.)
2. Splits within sections by paragraph boundaries
3. Keeps tables as atomic chunks (never split mid-row)
4. Overlap is paragraph-aligned and never crosses section boundaries
5. Each chunk gets a metadata prefix: `[Ticker: AAPL] [Year: 2024] [Section: Item 7. MD&A]`

**Parameters:** chunk_size=768 tokens, chunk_overlap=128 tokens, measured via tiktoken cl100k_base.

**Tradeoff:** Some table chunks exceed the target token count since they are never split. This is intentional — a truncated balance sheet is worse than an oversized chunk.

### 3. CRAG Self-Correction

**Problem:** Standard RAG fails silently. When retrieved context is irrelevant, the LLM generates plausible-sounding hallucinations from noise, with no signal to the user that the answer is ungrounded.

**Solution:** Corrective RAG adds a confidence gate:

1. Retrieve context for the original query
2. Ask the LLM to score context relevance on [0, 1]
3. If score < 0.6, reformulate the query (make it more SEC-specific) and re-retrieve
4. Generate the final answer from the best available context

**Tradeoff:** Adds 1 extra LLM call per query (relevance scoring), and 2 more when reformulation triggers. This roughly doubles latency on low-confidence queries. Acceptable because correctness matters more than speed for financial analysis.

### 4. Multi-Agent Memo Generation

**Problem:** Investment memos require both quantitative precision (exact XBRL numbers, computed ratios) and qualitative judgment (risk synthesis, outlook interpretation). A single prompt cannot reliably handle both in one pass.

**Solution:** Three specialized agents in sequence:

1. **Financial Data Agent** — Looks up XBRL metrics (revenue, net income, assets, liabilities, EPS, shares outstanding), computes ratios (gross margin, net margin, D/E, current ratio), formats for the LLM
2. **Qualitative Analysis Agent** — Three retrieval sub-queries for company overview (Item 1), risk factors (Item 1A), and MD&A (Item 7), each synthesized by the LLM
3. **Synthesis Agent** — Combines both outputs into executive summary and bull/bear cases

**Tradeoff:** 7 LLM calls per memo (company name + 6 section generations). Total latency ~30-60s depending on model. Could be parallelized (agents 1 and 2 are independent) but sequential execution is simpler to debug and log.

---

## Performance Characteristics

### Latency Breakdown (per query)

| Stage | Typical Latency | Notes |
|---|---|---|
| Dense search (ChromaDB) | 50-100ms | Local cosine HNSW, ~10K chunks |
| BM25 search | 10-30ms | In-memory numpy argsort |
| RRF fusion | <5ms | Dict merge + sort |
| Cross-encoder rerank | 500-1500ms | CPU inference, 10 candidate pairs |
| LLM relevance eval | 500-1000ms | Single GPT-4o-mini call |
| LLM answer generation | 1000-3000ms | Single GPT-4o-mini call |
| **Total (high confidence)** | **~2-5s** | No reformulation needed |
| **Total (low confidence)** | **~4-8s** | Reformulation + re-retrieve |

### Latency Breakdown (memo generation)

| Stage | Typical Latency |
|---|---|
| XBRL lookups + ratio calc | <100ms (in-memory) |
| 3 retrieval sub-queries | ~1-2s |
| 7 LLM calls | ~20-40s |
| **Total** | **~25-45s** |

### Token Usage Per Query

| Component | Input Tokens | Output Tokens |
|---|---|---|
| Relevance evaluation | ~800 | ~5 |
| Query reformulation (if triggered) | ~200 | ~50 |
| Answer generation | ~3000 (context + prompt) | ~200-500 |
| **Total (typical)** | **~3800** | **~300** |

### Token Usage Per Memo

| Component | Input Tokens | Output Tokens |
|---|---|---|
| Company name resolution | ~30 | ~10 |
| Financial highlights | ~1500 | ~300 |
| Company overview | ~2000 | ~300 |
| Risk factors | ~2000 | ~300 |
| MD&A synthesis | ~2000 | ~300 |
| Executive summary | ~2500 | ~200 |
| Bull/bear cases | ~2500 | ~300 |
| **Total** | **~12,500** | **~1,700** |

---

## Cost Analysis

Pricing based on OpenAI rates as of early 2025.

### Embedding Costs (ingestion)

| Item | Calculation |
|---|---|
| Model | text-embedding-3-small ($0.02 / 1M tokens) |
| Tokens per 10-K | ~80,000-120,000 (after chunking with prefixes) |
| Cost per 10-K | ~$0.002 |
| 3 companies x 3 years | ~$0.018 |

### Query Costs

| Item | Calculation |
|---|---|
| Model | GPT-4o-mini ($0.15 / 1M input, $0.60 / 1M output) |
| Per query (high confidence) | ~3,800 input + 300 output = $0.00075 |
| Per query (low confidence) | ~7,600 input + 600 output = $0.0015 |
| Per memo | ~12,500 input + 1,700 output = $0.0029 |

### Monthly Estimate

| Scenario | Queries/mo | Memos/mo | Estimated Cost |
|---|---|---|---|
| Development / demo | 100 | 10 | ~$0.10 |
| Light usage | 500 | 50 | ~$0.52 |
| Moderate usage | 2,000 | 200 | ~$2.08 |

Embedding costs are negligible after initial ingestion. Costs scale linearly with query volume. At $0.001/query, the system is viable for prototype and demo usage on a free-tier OpenAI account.

---

## Evaluation Framework

### Standard RAGAS Metrics

- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are the retrieved passages relevant?
- **Context Recall** — Does retrieval capture the ground-truth source sections?

### Custom Financial Metrics

- **Citation Accuracy** — Fraction of retrieved contexts matching ground-truth filing sections (string match on section identifiers)
- **Numerical Accuracy** — For numerical questions: fraction of ground-truth numbers matched within 1% tolerance. Handles financial formatting ($, commas, billion/million suffixes)

### Test Set

50 curated questions across 3 categories:
- 20 numerical/factual (exact XBRL-verifiable answers)
- 15 comparative (cross-company or cross-year)
- 15 analytical (risk factors, strategy, outlook)

Ground truth derived from actual AAPL, MSFT, and GOOGL 10-K filings.
