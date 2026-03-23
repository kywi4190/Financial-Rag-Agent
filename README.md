# Financial Document RAG Agent

Agentic RAG system that ingests SEC 10-K/10-Q filings, retrieves context via hybrid search, and generates cited investment memos with XBRL-grounded numerical accuracy.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)

[Live Demo](https://your-app.streamlit.app)

---

## Architecture

```
                           ┌─────────────────────────────────────────┐
                           │            Streamlit UI                  │
                           │  ┌───────────┬──────────┬────────────┐  │
                           │  │  Chat Q&A │  Memo    │  Eval      │  │
                           │  │  Tab      │  Tab     │  Dashboard │  │
                           │  └─────┬─────┴────┬─────┴──────┬─────┘  │
                           └────────┼──────────┼────────────┼────────┘
                                    │          │            │
                  ┌─────────────────▼──────────▼────────────┘
                  │          Agent Layer
                  │  ┌──────────────────────────────┐
                  │  │  FinancialQueryEngine (CRAG)  │
                  │  │  ┌─────────┐  ┌────────────┐ │
                  │  │  │Retrieve │→ │Evaluate    │ │
                  │  │  │Context  │  │Confidence  │ │
                  │  │  └─────────┘  └─────┬──────┘ │
                  │  │       ↑   < 0.5?    │        │
                  │  │       └─Reformulate─┘        │
                  │  └──────────────────────────────┘
                  │  ┌──────────────────────────────┐
                  │  │  MemoGenerator (3 Agents)     │
                  │  │  1. Financial Data → XBRL     │
                  │  │  2. Qualitative  → Retrieval  │
                  │  │  3. Synthesis    → LLM        │
                  │  └──────────────────────────────┘
                  └───────────────┬──────────────────
                                  │
                  ┌───────────────▼──────────────────┐
                  │       Hybrid Retrieval Pipeline    │
                  │                                    │
                  │  ┌──────────┐    ┌──────────┐     │
                  │  │  Dense   │    │  BM25    │     │
                  │  │  Vector  │    │  Sparse  │     │
                  │  │  Search  │    │  Search  │     │
                  │  └────┬─────┘    └────┬─────┘     │
                  │       │    ┌───┐      │           │
                  │       └───→│RRF│←─────┘           │
                  │            └─┬─┘                   │
                  │              ▼                     │
                  │  ┌──────────────────────┐         │
                  │  │  Cross-Encoder       │         │
                  │  │  Reranker            │         │
                  │  │  (ms-marco-MiniLM)   │         │
                  │  └──────────────────────┘         │
                  └───────────────┬──────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                    │
    ┌─────────▼────────┐  ┌──────▼───────┐  ┌────────▼───────┐
    │  ChromaDB        │  │  BM25 Index  │  │  XBRL Store    │
    │  (cosine HNSW)   │  │  (in-memory) │  │  (DataFrames)  │
    └─────────▲────────┘  └──────▲───────┘  └────────▲───────┘
              │                   │                    │
              └───────────────────┼───────────────────┘
                                  │
                  ┌───────────────▼──────────────────┐
                  │  Structure-Aware Chunker          │
                  │  • Section boundary preservation  │
                  │  • Atomic table chunks            │
                  │  • Metadata prefix injection      │
                  │  • tiktoken token counting        │
                  └───────────────┬──────────────────┘
                                  │
                  ┌───────────────▼──────────────────┐
                  │  Filing Parser                    │
                  │  • Item header extraction         │
                  │  • HTML/Markdown section parsing  │
                  │  • XBRL fact extraction           │
                  │  • Table detection + conversion   │
                  └───────────────┬──────────────────┘
                                  │
                  ┌───────────────▼──────────────────┐
                  │  SEC EDGAR Client (edgartools)    │
                  │  10-K / 10-Q download             │
                  └──────────────────────────────────┘
```

## Key Features

- **Hybrid retrieval** -- BM25 sparse + dense vector search fused via Reciprocal Rank Fusion, then reranked with a cross-encoder. Numerical queries automatically boost table/XBRL chunks.
- **Structure-aware chunking** -- Respects section boundaries, preserves financial tables as atomic chunks, and injects metadata prefixes (ticker, year, section) for grounded retrieval.
- **CRAG self-correction** -- Evaluates retrieval confidence; if below threshold (0.5), automatically reformulates the query and re-retrieves, keeping whichever result scores higher. Includes post-generation grounding verification that flags unsupported claims.
- **Multi-agent investment memos** -- Three specialized agents (Financial Data, Qualitative Analysis, Synthesis) produce structured memos with executive summary, risk factors, MD&A, and bull/bear cases.
- **XBRL-powered numerical accuracy** -- Extracts structured financial facts from inline XBRL, stores in typed DataFrames, and routes numerical queries to XBRL lookup tools before falling back to narrative text.
- **Citation tracking** -- Every generated claim carries a `Citation` object linking back to source document, section, ticker, year, and verbatim quote snippet.
- **RAGAS evaluation** -- Standard metrics (faithfulness, context precision/recall, answer relevancy) plus custom financial-domain metrics (citation accuracy, numerical accuracy with 1% tolerance).

## Evaluation Results

| Metric | Score | Notes |
|---|---|---|
| Faithfulness | **0.64** | CRAG self-correction and grounding verification reduce hallucination; analytical queries (0.46) lag behind comparative (0.78) due to longer answer chains |
| Context Precision | **0.40** | Cross-encoder reranker prioritizes relevant passages; analytical queries score well (0.86) but numerical (0.22) and comparative (0.18) suffer from sparse financial table context |
| Context Recall | **0.26** | Analytical queries achieve strong recall (0.74); numerical (0.05) and comparative (0.07) categories limited by balance sheet and multi-year data spanning multiple chunks |
| Answer Relevancy | **0.23** | Analytical queries score well (0.72); comparative queries return 0.0 due to multi-entity answers diverging from single-entity question embeddings |
| Citation Accuracy | **0.70** | Flexible ground-truth-oriented matching with component scoring (ticker, year, section); numerical queries strongest (0.83) |
| Numerical Accuracy | **0.34** | XBRL lookup handles direct fact retrieval; accuracy limited by derived metrics (margins, ratios) requiring multi-step calculation from raw XBRL data |

> Evaluated on 50 curated questions (20 numerical, 15 comparative, 15 analytical) across AAPL, MSFT, and GOOGL 10-K filings (2022-2025).

## Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Orchestration | LlamaIndex | Native tool-calling agents, FunctionTool bindings, prompt templates |
| Vector Store | ChromaDB | Persistent local storage, cosine HNSW index, metadata filtering |
| Embeddings | OpenAI `text-embedding-3-small` | Best cost/performance ratio at 1536 dims for financial text |
| LLM | GPT-4o-mini | Fast structured output for agent tool calls and memo generation |
| Sparse Search | rank-bm25 (Okapi BM25) | Captures exact keyword matches that embeddings miss (ticker symbols, XBRL concepts) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-12-v2 | Cross-encoder reranker that fits in CPU memory, strong passage ranking accuracy |
| SEC Data | edgartools | Typed Python API for EDGAR with markdown/HTML export and XBRL support |
| Tokenization | tiktoken (cl100k_base) | Exact token counting matching the OpenAI embedding model's tokenizer |
| Data Models | Pydantic v2 | Strict type validation on all pipeline data, JSON serialization for eval |
| Evaluation | RAGAS + custom metrics | Industry-standard RAG evaluation plus financial-domain numerical accuracy |
| Frontend | Streamlit | Rapid prototyping with built-in chat UI, tabs, and chart components |

## Design Decisions

### Hybrid Search Over Vector-Only

Dense vector search alone struggles with financial documents. Ticker symbols, XBRL concept names like `us-gaap:Revenues`, and exact dollar amounts are lexically significant but don't embed well into semantic space. BM25 catches these exact matches. Reciprocal Rank Fusion (RRF) with k=60 merges the two ranked lists, weighting dense at 0.7 and sparse at 0.3. For numerical queries (detected via pattern matching), table and XBRL chunks receive a 1.5x score boost. The cross-encoder reranker then re-scores the top candidates using full query-passage attention, producing a final top-5 that balances semantic understanding with keyword precision.

### Structure-Aware Chunking Over Fixed-Size

SEC filings have strong structural signals -- Item headers, section boundaries, financial tables -- that naive fixed-size chunking destroys. The chunker first splits by filing section (Item 1, Item 1A, Item 7, etc.), then by paragraph boundaries within each section. Overlap never crosses section boundaries, preventing context pollution between unrelated topics. Financial tables are always kept as atomic chunks and never split mid-row, preserving the column relationships that make a balance sheet or income statement readable. Each chunk is prefixed with `[Ticker: AAPL] [Year: 2024] [Section: Item 7. MD&A]` metadata, giving both the embedding model and the LLM grounding context at retrieval time.

### CRAG Self-Correction Loop

Retrieval-Augmented Generation fails silently when retrieved context is irrelevant -- the LLM hallucinates a plausible-sounding answer from noise. Corrective RAG (CRAG) adds a confidence evaluation step: after initial retrieval, the LLM scores context relevance on a 0-1 scale. If the score falls below 0.5, the system reformulates the query (making it more specific to SEC filing terminology) and re-retrieves, keeping whichever result scores higher. A post-generation grounding verification step then checks whether the answer is actually supported by the retrieved context, flagging unsupported claims. This catches cases where a user's natural language doesn't match the formal language of filings -- e.g., "How much cash does Apple have?" reformulated to target "cash and cash equivalents" in Item 8 financial statements.

### Multi-Agent Memo Generation Over Single-Prompt

Investment memos require both quantitative precision and qualitative judgment -- a single prompt can't reliably handle both. The system decomposes memo generation into three specialized agents: (1) a Financial Data Agent that extracts XBRL metrics and computes ratios (gross margin, D/E, current ratio), producing grounded numbers; (2) a Qualitative Analysis Agent that retrieves and synthesizes risk factors, MD&A commentary, and business descriptions from narrative sections; (3) a Synthesis Agent that combines both outputs into an executive summary and bull/bear cases. Each agent operates on its own context window with task-specific prompts, avoiding the context dilution that degrades single-prompt quality on long-form financial analysis.

## Quick Start

```bash
git clone https://github.com/yourusername/Financial-Rag-Agent.git
cd Financial-Rag-Agent

pip install -r requirements.txt

# Add your API keys to .env
echo "OPENAI_API_KEY=sk-..." > .env
echo "SEC_EDGAR_IDENTITY=YourName your@email.com" >> .env

python scripts/ingest.py --ticker AAPL

streamlit run app/main.py
```

## Project Structure

```
Financial-Rag-Agent/
├── app/
│   ├── main.py                    # Streamlit entry point
│   └── components/
│       ├── chat.py                # Q&A chat interface with citations
│       ├── memo.py                # Investment memo generation tab
│       └── metrics.py             # RAGAS eval dashboard
├── src/
│   ├── config.py                  # Pydantic settings from .env
│   ├── ingestion/
│   │   ├── edgar_client.py        # SEC EDGAR download client
│   │   ├── filing_parser.py       # Section + XBRL extraction
│   │   └── models.py             # FilingMetadata, FilingSection, XBRLFact
│   ├── chunking/
│   │   ├── financial_chunker.py   # Structure-aware chunking
│   │   ├── table_handler.py       # HTML table → markdown conversion
│   │   └── models.py             # ChunkMetadata, DocumentChunk
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB wrapper with batched embedding
│   │   ├── bm25_search.py         # Okapi BM25 sparse index
│   │   ├── hybrid.py             # RRF fusion + numerical query boosting
│   │   ├── reranker.py           # Cross-encoder reranking
│   │   └── models.py             # SearchResult, RetrievalConfig
│   ├── agents/
│   │   ├── query_engine.py        # CRAG query engine
│   │   ├── financial_tools.py     # XBRL lookup, ratio calc, comparison tools
│   │   ├── memo_generator.py      # Multi-agent memo orchestration
│   │   └── models.py             # Citation, InvestmentMemo, FinancialCalc
│   └── evaluation/
│       ├── ragas_eval.py          # RAGAS + custom financial metrics
│       ├── test_questions.py      # 50 curated eval questions
│       └── models.py             # EvalQuestion, EvalReport
├── scripts/
│   ├── ingest.py                  # CLI: download → parse → chunk → index
│   └── evaluate.py                # CLI: run RAGAS evaluation
├── tests/                         # Pytest suite mirroring src/
├── requirements.txt
└── CLAUDE.md
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Point it at your repo, branch `main`, main file path `app/main.py`.
4. Open **Advanced settings > Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-..."
   SEC_EDGAR_IDENTITY = "YourName your@email.com"
   ```
5. Click **Deploy**. The app will install from `requirements.txt` and `packages.txt` automatically.

**Pre-configured files:**

| File | Purpose |
|---|---|
| `.streamlit/config.toml` | Professional theme, XSRF protection, usage stats opt-out |
| `.streamlit/secrets.toml.example` | Template for required secrets |
| `packages.txt` | System-level apt packages (`build-essential` for native extensions) |

> The app detects missing API keys at startup and displays setup instructions instead of crashing.

## Known Limitations & Future Work

**Current limitations:**
- XBRL extraction limited to inline `ix:nonfraction` tags; does not parse full XBRL instance documents
- Reranker runs on CPU -- adds ~2-5s latency on the cross-encoder pass per query
- Evaluation requires ingested data and API calls -- no offline/cached mode
- Numerical and comparative queries have near-zero context recall (0.05 and 0.07), indicating retrieval struggles with balance sheet lookups and multi-entity comparisons
- Comparative answer relevancy scores 0.0 due to multi-entity answers diverging from single-entity question embeddings in the similarity metric

**Evaluation-informed improvements** (scores below 0.7):
- **Context Recall (0.26):** Analytical queries achieve 0.74 recall, but numerical (0.05) and comparative (0.07) categories fail to retrieve the right chunks — metadata filtering needs to better match fiscal year queries to filing dates, and financial tables need to be chunked at the statement level rather than row level
- **Context Precision (0.40):** Analytical queries score 0.86, showing the cross-encoder reranker works well for narrative text — numerical and comparative queries need section-aware boosting tuned specifically for financial statement chunks
- **Faithfulness (0.64):** Comparative queries score 0.78 but analytical queries drop to 0.46 — longer analytical answers accumulate more ungrounded claims; tighter grounding verification or per-claim citation enforcement could help
- **Answer Relevancy (0.23):** Comparative category returns 0.0 because the embedding similarity metric penalizes multi-entity answers — needs a domain-specific relevancy metric or query-type-aware evaluation
- **Numerical Accuracy (0.34):** Direct XBRL fact lookups work but derived metrics (margins, ratios, YoY changes) require multi-step calculation that the current single-tool pipeline doesn't support — needs a computation chain that extracts components then calculates

**Planned improvements:**
- Full XBRL instance document parsing for broader fact coverage
- Async parallel retrieval (dense + sparse concurrently)
- Streaming LLM responses in the Streamlit chat interface
- Multi-filing temporal analysis (auto-compare across fiscal years)
- GPU-accelerated reranking for production latency targets
- Multi-hop reasoning for complex comparative questions requiring data from multiple filings
- Computation chain for derived financial metrics (extract XBRL components, then calculate ratios/margins)
- Statement-level table chunking to keep full balance sheets and income statements intact
- Domain-specific evaluation metrics that handle multi-entity comparative answers

---

Built by Kyle Wilson -- finance background + applied math + software engineering.
