# Phase 2: Performance & Reliability — Claude Code Prompt Guide

> **Goal**: Improve foundational retrieval quality, fix the RAGAS embedding compatibility issue, and boost all evaluation metrics below 0.7. Infrastructure reliability improvements where they matter most.
>
> **Baseline scores**: Faithfulness 0.60 | Context Precision 0.49 | Context Recall 0.35 | Answer Relevancy N/A | Citation Accuracy 0.54 | Numerical Accuracy 0.72

---

## How to Use This Guide

1. **Execute prompts in order.** Each prompt depends on changes from prior prompts. The ordering is designed so that foundational retrieval improvements compound into downstream score gains.

2. **Clear context (`/clear`) before each prompt.** This is critical — Claude Code performs best with a fresh context window focused on one task. Each prompt is self-contained with all the context it needs.

3. **Use `ultrathink` as indicated.** Prompts 1–5 begin with `ultrathink` because they involve complex multi-file reasoning. Prompts 6–7 do not need it.

4. **Verify before moving on.** Each prompt includes a test command. If tests fail, fix them in the same session before clearing context. If a commit instruction says to commit, do so before clearing.

5. **Budget for API costs.** Prompt 7 (final evaluation) runs 50 LLM queries through the full pipeline. Expect ~$2–5 in OpenAI API costs depending on your model.

6. **Estimated total execution time**: 2–4 hours across all 7 prompts (excluding evaluation API time).

---

## Prompt 1 — Fix RAGAS Embedding Compatibility

> **Unblocks**: Answer Relevancy metric (currently N/A)
> **Files to modify**: `src/evaluation/ragas_eval.py`, `tests/test_evaluation.py`

### Clear context, then paste:

```
ultrathink

## Context
I have a Financial RAG Agent project. The RAGAS `answer_relevancy` metric fails because it
expects a langchain-compatible embedding object with an `embed_query()` method. The project uses
LlamaIndex's `OpenAIEmbedding` which only exposes `get_text_embedding()` and
`get_text_embedding_batch()`. This means `answer_relevancy` has been returning N/A in all
evaluation runs.

## Current Code

### `src/evaluation/ragas_eval.py`
The `_compute_ragas_metrics` method (line 128) calls:
```python
ragas_result = ragas_evaluate(dataset, metrics=metrics)
```
No embedding model is passed, so RAGAS tries to use its default, which fails with LlamaIndex.

The RAGAS dataset uses 0.4.x column names:
```python
dataset = Dataset.from_dict({
    "user_input": [...],
    "response": [...],
    "retrieved_contexts": [...],
    "reference": [...],
})
```

The evaluator constructor stores `self._llm_model` (default "gpt-4o-mini") but has no embedding
config.

### `requirements.txt`
Includes: `ragas>=0.1.0`, `llama-index-embeddings-openai>=0.1.0`, `openai>=1.0.0`

## Task

### 1. Create an embedding adapter class (in `src/evaluation/ragas_eval.py`, NOT a new file)
Add a `LlamaIndexEmbeddingAdapter` class that:
- Wraps `llama_index.embeddings.openai.OpenAIEmbedding`
- Exposes `embed_query(text: str) -> list[float]` — delegates to `get_text_embedding()`
- Exposes `embed_documents(texts: list[str]) -> list[list[float]]` — delegates to
  `get_text_embedding_batch()`
- Type hints on all signatures, Google-style docstring

### 2. Update `_compute_ragas_metrics` to use the adapter
- Create a `LlamaIndexEmbeddingAdapter` instance using the project's configured embedding model
  (get it from `src.config.get_settings().embedding_model`)
- Pass the adapter to `ragas_evaluate()` via the `embeddings` parameter
- If the ragas version uses a different parameter name for embeddings (e.g., `llm_factory`,
  `RunConfig`, or `EvaluatorConfig`), investigate the installed ragas API and use the correct
  approach. The key is making `answer_relevancy` receive an embedding object with `embed_query()`.

### 3. Update `RAGASEvaluator.__init__` to accept an optional `embedding_model` parameter
- Default to `None`, and if `None`, read from `get_settings().embedding_model`
- Store as `self._embedding_model`

### 4. Add tests in `tests/test_evaluation.py`
- Test that `LlamaIndexEmbeddingAdapter.embed_query()` returns a list of floats
- Test that `LlamaIndexEmbeddingAdapter.embed_documents()` returns list of list of floats
- Mock the underlying `OpenAIEmbedding` to avoid real API calls (mock `get_text_embedding` to
  return a list of 10 floats, mock `get_text_embedding_batch` to return a list of lists)

### Rules
- Do NOT modify any files outside `src/evaluation/ragas_eval.py` and `tests/test_evaluation.py`
- Follow the project's code style: type hints on ALL signatures, Pydantic models for structured
  data, Google-style docstrings, pathlib not os.path
- Run `python -m pytest tests/test_evaluation.py -v` after changes and fix any failures
- Commit with a descriptive message. Do NOT include Co-Authored-By in the commit message.
```

---

## Prompt 2 — Improve Context Recall

> **Target**: Context Recall 0.35 → 0.50+
> **Files to modify**: `src/agents/query_engine.py`, `src/agents/financial_tools.py`, `src/retrieval/hybrid.py`, `src/retrieval/bm25_search.py`, `tests/test_retrieval.py`, `tests/test_agents.py`

### Clear context, then paste:

```
ultrathink

## Context
I have a Financial RAG Agent project. Context Recall is the worst metric at 0.35. Three root
causes:

1. **Retrieval depth too shallow**: The query engine and context retrieval tool both default to
   `top_k=5`, even though the config has `top_k=10`. Relevant chunks are retrieved but discarded.
2. **No query expansion**: Financial documents use varied terminology (e.g., "revenue" vs "net
   sales" vs "total revenue"). Queries miss relevant chunks that use synonyms.
3. **Poor BM25 tokenization**: The BM25 tokenizer splits financial terms like "10-K", "$391",
   "46.2%", and "year-over-year" into meaningless fragments.

## Current Code — What Needs to Change

### `src/agents/query_engine.py` (line 113-127)
```python
def _retrieve_context(
    self, query: str, ticker: str | None = None, top_k: int = 5,
) -> list[dict[str, Any]]:
```
The `top_k=5` default should be `top_k=10` to match `Settings.top_k=10` in config.py.

### `src/agents/financial_tools.py` (line 279-284)
```python
def retrieve_context_tool(
    query: str, retriever: HybridRetriever,
    ticker: str | None = None, top_k: int = 5,
) -> list[dict[str, Any]]:
```
Same issue — `top_k=5` should be `top_k=10`.

### `src/retrieval/hybrid.py` (HybridRetriever.search)
Currently just does: dense search → BM25 search → RRF fusion → numerical boost → return.
No query expansion happens anywhere.

### `src/retrieval/bm25_search.py` (line 31-34)
```python
@staticmethod
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
```
This regex splits "10-K" into ["10", "k"], "$391.0" into ["391", "0"], etc.

## Task

### 1. Increase retrieval depth
- In `src/agents/query_engine.py`: change `_retrieve_context` default `top_k` from 5 to 10
- In `src/agents/financial_tools.py`: change `retrieve_context_tool` default `top_k` from 5 to 10

### 2. Add financial synonym query expansion (`src/retrieval/hybrid.py`)
Add a `_expand_query` static method to `HybridRetriever`:
```python
_FINANCIAL_SYNONYMS: dict[str, list[str]] = {
    "revenue": ["net sales", "total revenue", "net revenue"],
    "earnings": ["net income", "net earnings", "profit"],
    "profit": ["net income", "operating income", "gross profit"],
    "debt": ["liabilities", "borrowings", "long-term debt"],
    "assets": ["total assets", "current assets"],
    "cash flow": ["operating cash flow", "cash from operations", "cash generated by operating"],
    "eps": ["earnings per share"],
    "r&d": ["research and development"],
    "sg&a": ["selling general and administrative"],
    "capex": ["capital expenditures"],
    "stockholders equity": ["shareholders equity", "total equity"],
    "cost of revenue": ["cost of goods sold", "cogs"],
}
```
The method should:
- Scan the query (case-insensitive) for keys in the synonym map
- Append matching synonym terms to the query string, separated by " OR "
- Example: "What was Apple's revenue?" → "What was Apple's revenue? OR net sales OR total revenue OR net revenue"
- Call `_expand_query` in `search()` BEFORE passing the query to both dense and BM25 search

### 3. Improve BM25 tokenizer (`src/retrieval/bm25_search.py`)
Replace the `_tokenize` method with one that preserves financial compound terms:
- Keep "10-K" and "10-Q" as single tokens
- Keep dollar amounts (e.g., "$391", "$391.0") as single tokens (strip the $)
- Keep percentages (e.g., "46.2%") as single tokens (strip the %)
- Keep hyphenated compound words (e.g., "year-over-year", "long-term") as single tokens
- Still lowercase everything
- Still split on whitespace and most punctuation, just with smarter patterns

### Tests
Update `tests/test_retrieval.py`:
- Add test for `HybridRetriever._expand_query`:
  - Test query with "revenue" → expanded query contains "net sales"
  - Test query with no matching terms → returns unchanged
  - Test query with multiple matching terms
- Add test for improved BM25 tokenizer:
  - Test "10-K" stays as single token
  - Test "$391.0 billion" → tokens include "391.0" and "billion"
  - Test "year-over-year" stays as single token

Update `tests/test_agents.py`:
- If any test hardcodes `top_k=5` in assertions or mock expectations, update to `top_k=10`

Run: `python -m pytest tests/test_retrieval.py tests/test_agents.py -v`

### Rules
- Do NOT change `RetrievalConfig.top_k` (it's 20, controlling how many candidates each source
  returns before fusion — that's correct)
- Do NOT integrate the Reranker yet (that's the next prompt)
- The synonym expansion should be additive — never replace the original query terms
- Commit with a descriptive message. Do NOT include Co-Authored-By.
```

---

## Prompt 3 — Improve Context Precision

> **Target**: Context Precision 0.49 → 0.60+
> **Files to modify**: `src/retrieval/hybrid.py`, `scripts/evaluate.py`, `app/main.py`, `tests/test_retrieval.py`

### Clear context, then paste:

```
ultrathink

## Context
I have a Financial RAG Agent project. Context Precision is 0.49. Two major issues:

1. **Reranker not integrated**: A fully implemented `Reranker` class exists at
   `src/retrieval/reranker.py` with a `rerank(query, results, top_k)` method using
   `cross-encoder/ms-marco-MiniLM-L-6-v2`. It's tested but NEVER imported or used in the actual
   pipeline. The `HybridRetriever.search()` returns RRF-fused results without reranking.

2. **No query-type-aware retrieval**: All queries are treated identically. A balance sheet question
   should prioritize Item 8 chunks, a risk question should prioritize Item 1A, etc.

## Current Architecture

### `src/retrieval/reranker.py` — exists but unused
```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=5) -> list[SearchResult]:
        # Predicts cross-encoder scores, min-max normalizes to [0,1], returns top_k
```

### `src/retrieval/hybrid.py` — HybridRetriever.search()
Current pipeline: dense → BM25 → RRF fusion → numerical boost → return `fused[:cfg.top_k]`
The `RetrievalConfig` already has `rerank_top_k: int = 5` but it's never used.

### `src/retrieval/models.py` — RetrievalConfig
```python
class RetrievalConfig(BaseModel):
    top_k: int = 20
    rerank_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    filters: Optional[dict] = None
```

### `scripts/evaluate.py` (line 195)
```python
retriever = HybridRetriever(vector_store, bm25, RetrievalConfig())
```
No reranker passed.

### `app/main.py` (line 84)
```python
return HybridRetriever(_vector_store, bm25, config)
```
No reranker passed.

### Chunk metadata
Each chunk has `metadata.section_name` like "Item 1. Business", "Item 1A. Risk Factors",
"Item 7. MD&A", "Item 8. Financial Statements", etc.

## Task

### 1. Integrate Reranker into HybridRetriever (`src/retrieval/hybrid.py`)
- Add `from src.retrieval.reranker import Reranker` import
- Add optional `reranker: Reranker | None = None` parameter to `__init__`
- Store as `self._reranker`
- In `search()`, after RRF fusion and numerical boosting:
  - If `self._reranker` is not None: `reranked = self._reranker.rerank(query, fused, cfg.rerank_top_k)` then return `reranked`
  - If `self._reranker` is None: return `fused[:cfg.top_k]` (current behavior, backward compat)
- The pipeline becomes: dense + BM25 → RRF fusion → numerical boost → cross-encoder rerank → top_k

### 2. Add query-type section boosting (`src/retrieval/hybrid.py`)
Add a `_detect_target_sections` static method:
```python
_SECTION_ROUTING: dict[str, list[str]] = {
    r"balance sheet|assets|liabilities|equity|stockholders": ["Item 8"],
    r"risk|risk factor": ["Item 1A"],
    r"revenue|income|margin|earnings|profit|operat": ["Item 7", "Item 8"],
    r"business|overview|strategy|product|segment": ["Item 1"],
    r"market risk|interest rate|currency|foreign exchange": ["Item 7A"],
    r"cash flow|liquidity|capital": ["Item 7", "Item 8"],
}
```
- Method takes a query string, returns list of target section prefixes (e.g., ["Item 7", "Item 8"])
- Match using regex (case-insensitive), return union of all matching sections
- In `search()`, after reranking (or after numerical boost if no reranker), apply a 1.3x soft
  boost to results whose `metadata.section_name` starts with any target section prefix
- Re-sort by score after boosting
- This is a soft boost, not a hard filter — non-matching sections can still rank highly

### 3. Update pipeline initialization
**`scripts/evaluate.py`** (around line 195):
- Import Reranker: `from src.retrieval.reranker import Reranker`
- Create reranker: `reranker = Reranker()`
- Pass to HybridRetriever: `retriever = HybridRetriever(vector_store, bm25, RetrievalConfig(), reranker=reranker)`

**`app/main.py`** (`_init_retriever` function around line 56):
- Same pattern: import Reranker, create instance, pass to HybridRetriever
- Wrap Reranker initialization in try/except (model download may fail on first run)

### Tests (`tests/test_retrieval.py`)
- Test HybridRetriever with a mock reranker calls `rerank()`
- Test HybridRetriever without reranker (None) still works unchanged
- Test `_detect_target_sections` for:
  - "balance sheet" query → returns sections containing "Item 8"
  - "risk factors" query → returns sections containing "Item 1A"
  - "revenue" query → returns sections containing "Item 7" and "Item 8"
  - generic query with no matches → returns empty list
- Test section boost: result with matching section gets higher score than non-matching

Run: `python -m pytest tests/test_retrieval.py -v`

### Rules
- Reranker is already fully tested — don't duplicate those tests
- The reranker import should use a try/except in app/main.py for graceful degradation
- Commit with a descriptive message. Do NOT include Co-Authored-By.
```

---

## Prompt 4 — Improve Citation Accuracy

> **Target**: Citation Accuracy 0.54 → 0.70+
> **Files to modify**: `src/evaluation/ragas_eval.py`, `tests/test_evaluation.py`

### Clear context, then paste:

```
ultrathink

## Context
I have a Financial RAG Agent project. Citation Accuracy is 0.54. The root cause is a mismatch
between how ground truth contexts are formatted and how the evaluation matches them against
retrieved contexts.

## The Mismatch in Detail

### Ground truth format (in `src/evaluation/test_questions.py`)
```python
ground_truth_contexts=["AAPL 10-K 2024, Item 8. Financial Statements"]
```
Format: `"{ticker} {filing_type} {year}, {section_name}"`

### Chunk content format (what's actually retrieved)
Chunks have a metadata prefix in their content:
```
[Ticker: AAPL] [Year: 2024] [Section: Item 8. Financial Statements] [Filing: 10-K]
```

### How the evaluator collects contexts (`src/evaluation/ragas_eval.py` lines 106-119)
```python
raw_contexts = query_engine._retrieve_context(question.question)
contexts = [c["content"] for c in raw_contexts]
```
It only gets raw content strings. The content DOES contain the metadata prefix, so section names
like "Item 8" are present as substrings. But the full ground truth string "AAPL 10-K 2024" does
NOT appear in bracket format.

### Current matching logic (`_compute_citation_accuracy`, lines 242-276)
```python
for ctx in result.retrieved_contexts:
    ctx_lower = ctx.lower()
    for gt_section in question.ground_truth_contexts:
        if gt_section.lower() in ctx_lower:  # Full string substring match
            matches += 1
            break
        section_parts = gt_section.split(", ")
        if len(section_parts) >= 2:
            section_name = section_parts[-1].lower()
            if section_name in ctx_lower:  # Section name only
                matches += 1
                break
```
The full string match rarely works because of format differences. The section-only fallback works
but is too loose (matches wrong company's chunks if they share section names).

The score = matches / len(retrieved_contexts), so false matches from wrong companies inflate
incorrectly, and missed matches on correct company deflate the score.

## Task

### 1. Improve citation matching logic (`src/evaluation/ragas_eval.py`)
Replace `_compute_citation_accuracy` with component-based matching:

```python
@staticmethod
def _parse_ground_truth_context(gt_context: str) -> dict[str, str] | None:
    """Parse 'AAPL 10-K 2024, Item 8. Financial Statements' into components."""
    # Expected format: "{ticker} {filing_type} {year}, {section}"
    # Return dict with keys: ticker, filing_type, year, section
    # Return None if format doesn't match (fall back to substring matching)
```

Update `_compute_citation_accuracy` to:
- Parse each ground_truth_context into components using `_parse_ground_truth_context`
- For each retrieved context, check component matches:
  - ticker match: the ticker appears in the context (word boundary or in bracket format)
  - section match: the section name appears in the context (case-insensitive substring)
  - year match: the year appears in the context (substring)
- Scoring per retrieved context:
  - All three match → 1.0 (full match)
  - Section + year match (but wrong/missing ticker) → 0.5 (partial)
  - Only section matches → 0.25 (weak partial)
  - No section match → 0.0
- Final score: sum of per-context scores / len(retrieved_contexts)
- If `_parse_ground_truth_context` returns None, fall back to the original substring matching

### 2. Enrich context strings with citation metadata (`src/evaluation/ragas_eval.py`)
In `_run_single_question` (around line 110), change:
```python
contexts = [c["content"] for c in raw_contexts]
```
to:
```python
contexts = [
    f"{c['citation']['source_document']}, {c['citation']['section']}: {c['content']}"
    for c in raw_contexts
]
```
This prepends citation info like `"AAPL 2024 10-K, Item 8. Financial Statements: [Ticker: ...]"`
to each context string, making component matching much more reliable since the citation metadata
is guaranteed to contain the ticker, year, and section.

### Tests (`tests/test_evaluation.py`)
Add/update these tests:

```python
class TestParseGroundTruthContext:
    def test_standard_format(self):
        result = RAGASEvaluator._parse_ground_truth_context(
            "AAPL 10-K 2024, Item 8. Financial Statements"
        )
        assert result == {
            "ticker": "AAPL", "filing_type": "10-K",
            "year": "2024", "section": "Item 8. Financial Statements"
        }

    def test_invalid_format_returns_none(self):
        result = RAGASEvaluator._parse_ground_truth_context("some random text")
        assert result is None
```

Update `TestCitationAccuracy`:
- Test full match (ticker + section + year all present) → scores 1.0
- Test partial match (section + year match, wrong ticker) → scores 0.5
- Test section-only match → scores 0.25
- Test no match → scores 0.0
- Test with enriched context format (citation metadata prepended)
- Test fallback behavior when ground truth format is unparseable

Run: `python -m pytest tests/test_evaluation.py -v`

### Rules
- Do NOT modify `src/evaluation/test_questions.py` or the ground truth data
- Do NOT modify chunk content or the chunker — only modify the evaluation pipeline
- Keep backward compatibility: unparseable ground truth contexts use original substring matching
- Commit with a descriptive message. Do NOT include Co-Authored-By.
```

---

## Prompt 5 — Improve Faithfulness

> **Target**: Faithfulness 0.60 → 0.70+
> **Files to modify**: `src/agents/query_engine.py`, `tests/test_agents.py`

### Clear context, then paste:

```
ultrathink

## Context
I have a Financial RAG Agent project. Faithfulness is 0.60 — the system sometimes generates
claims not grounded in retrieved context. Three root causes:

1. **CRAG threshold too low**: `CRAG_CONFIDENCE_THRESHOLD = 0.6` lets mediocre context through
2. **Context evaluation too narrow**: `_evaluate_context_relevance` only looks at top 3 chunks,
   500 chars each — misses important context in later chunks
3. **No post-generation grounding check**: The generation prompt says "based ONLY on context" but
   there's no verification that the answer actually follows this instruction

## Current Code in `src/agents/query_engine.py`

### Line 42
```python
CRAG_CONFIDENCE_THRESHOLD = 0.6
```

### Lines 148-149 in `_evaluate_context_relevance`
```python
context_text = "\n\n".join(
    c["content"][:500] for c in context_chunks[:3]
)
```

### Lines 220-226 in `_generate_answer` — the generation prompt
```python
prompt = (
    f"{FINANCIAL_SYSTEM_PROMPT}\n\n"
    f"Context from SEC filings:\n{context_text}\n\n"
    f"Question: {query}\n\n"
    f"Provide a thorough answer based ONLY on the context above. "
    f"Cite the source document and section for each claim."
)
```

### Lines 256-289 — the `query()` method (CRAG flow)
```python
def query(self, question, ticker=None) -> AnswerWithCitations:
    context = self._retrieve_context(question, ticker=ticker)
    confidence = self._evaluate_context_relevance(question, context)
    if confidence < CRAG_CONFIDENCE_THRESHOLD:
        reformulated = self._reformulate_query(question)
        context = self._retrieve_context(reformulated, ticker=ticker)
        confidence = self._evaluate_context_relevance(reformulated, context)
    answer = self._generate_answer(question, context)
    return answer
```

### AnswerWithCitations model (from `src/agents/models.py`)
```python
class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
```

## Task

### 1. Raise CRAG threshold (`src/agents/query_engine.py`)
- Change `CRAG_CONFIDENCE_THRESHOLD` from `0.6` to `0.7`
- This triggers reformulation more aggressively, catching more low-quality retrievals

### 2. Widen context evaluation window (`src/agents/query_engine.py`)
In `_evaluate_context_relevance`:
- Change `context_chunks[:3]` to `context_chunks[:5]` (evaluate more chunks)
- Change `c["content"][:500]` to `c["content"][:800]` (see more of each chunk)

### 3. Strengthen the generation prompt (`src/agents/query_engine.py`)
Update the prompt in `_generate_answer` to append these additional instructions:
```
- If the context does not contain specific information to answer any part of the question,
  explicitly state that the information is not available rather than speculating.
- Do not make inferences or calculations beyond what is directly stated in the context.
- Every numerical claim must reference the specific source.
```

### 4. Add post-generation grounding verification (`src/agents/query_engine.py`)
Add a new method:
```python
def _verify_grounding(
    self,
    answer_text: str,
    context_chunks: list[dict[str, Any]],
) -> tuple[str, float]:
    """Verify that the generated answer is grounded in the provided context.

    Args:
        answer_text: The generated answer to verify.
        context_chunks: The context chunks used for generation.

    Returns:
        Tuple of (verified_answer, grounding_score).
        If grounding is poor, returns a revised answer with only supported claims.
    """
```

Implementation:
- Build a context string from chunks (use first 5 chunks, 1000 chars each)
- Prompt the LLM: "Review this answer for faithfulness to the provided context. Score the
  grounding on a 0.0-1.0 scale. If any claims are NOT supported by the context, rewrite the
  answer to include only well-supported claims. Respond in this exact format:
  SCORE: <number>
  ANSWER: <revised or original answer>"
- Parse the response to extract the score and answer
- If parsing fails, return (original answer, 0.5) as fallback

### 5. Integrate grounding check into `query()` method
After `answer = self._generate_answer(question, context)`:
- Call `verified_answer, grounding_score = self._verify_grounding(answer.answer, context)`
- If `grounding_score < 0.5`: replace with the "insufficient information" response
  (confidence 0.0, empty citations)
- Otherwise: update `answer.answer` with `verified_answer` and set `answer.confidence` to
  `min(answer.confidence, grounding_score)`

### Tests (`tests/test_agents.py`)

**IMPORTANT**: The existing tests mock `self._llm.complete` with `side_effect` lists. Adding the
grounding verification adds one more LLM call to the `query()` flow. You need to add one more
mock response to each `side_effect` list in tests that call `query()`.

Update these tests:
- `test_crag_reformulates_on_low_confidence`: The threshold is now 0.7. If the test uses a mock
  confidence of 0.65, it should now trigger reformulation (previously it wouldn't at 0.6 threshold).
  Adjust mock responses. Also add a grounding verification response to the side_effect list.
- `test_query_returns_answer_with_citations`: Add grounding verification mock response.
  The mock should return something like "SCORE: 0.9\nANSWER: <same answer>".
- `test_empty_retrieval_returns_low_confidence`: This path may not reach grounding verification
  (no context → returns early). Verify this.

Add new test:
- `test_verify_grounding_low_score_returns_insufficient_data`: Mock the grounding LLM call to
  return a low score (e.g., 0.3). Assert that the final answer is the "insufficient information"
  fallback and confidence is 0.0.

Run: `python -m pytest tests/test_agents.py -v`

### Rules
- The grounding check adds one LLM call per query — this is an acceptable cost for quality
- Do NOT modify any files outside `src/agents/query_engine.py` and `tests/test_agents.py`
- Follow existing code style: type hints, Google-style docstrings
- Commit with a descriptive message. Do NOT include Co-Authored-By.
```

---

## Prompt 6 — BM25 Index Persistence

> **Resolves**: "BM25 index is rebuilt in-memory on each Streamlit startup" limitation
> **Files to modify**: `src/retrieval/bm25_search.py`, `src/config.py`, `scripts/evaluate.py`, `app/main.py`, `tests/test_retrieval.py`

### Clear context, then paste:

```
## Context
I have a Financial RAG Agent project. The BM25 index (`src/retrieval/bm25_search.py`) is rebuilt
in-memory from ChromaDB documents on every startup. This is slow for large corpora. I need to add
save/load persistence.

## Current BM25Index class (`src/retrieval/bm25_search.py`)
```python
class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # tokenization logic...

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [self._tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        # search logic...
```

## Current startup code

### `scripts/evaluate.py` (lines 182-193)
```python
bm25 = BM25Index()
all_docs = vector_store._collection.get(include=["documents", "metadatas"])
if all_docs["documents"]:
    chunks = []
    for i, doc in enumerate(all_docs["documents"]):
        meta = all_docs["metadatas"][i]
        chunks.append(DocumentChunk(
            chunk_id=all_docs["ids"][i], content=doc,
            metadata=ChunkMetadata(**meta),
        ))
    bm25.build_index(chunks)
```

### `app/main.py` (`_init_retriever`, lines 62-87)
Same pattern: creates BM25Index, loads all docs from ChromaDB, builds index.

## Task

### 1. Add persistence methods (`src/retrieval/bm25_search.py`)
Add `import pickle` and `from pathlib import Path` to imports.

Add `save_index` method:
```python
def save_index(self, path: Path) -> None:
    """Serialize the BM25 index and chunk data to disk.

    Args:
        path: File path to save the index (e.g., .bm25/bm25_index.pkl).
    """
```
- Save a dict containing: `{"chunks_data": [...], "tokenized_corpus": [...]}`
- For chunks_data, serialize each chunk as: `{"chunk_id": ..., "content": ..., "metadata": chunk.metadata.model_dump()}`
- Create parent directories if needed (`path.parent.mkdir(parents=True, exist_ok=True)`)
- Use `pickle.dump` with `protocol=pickle.HIGHEST_PROTOCOL`
- Log the save with chunk count

Add `load_index` classmethod:
```python
@classmethod
def load_index(cls, path: Path) -> "BM25Index":
    """Load a previously saved BM25 index from disk.

    Args:
        path: Path to the saved index file.

    Returns:
        Restored BM25Index ready for search.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
```
- Load with `pickle.load`
- Reconstruct `DocumentChunk` objects from saved data
- Rebuild `BM25Okapi` from the saved tokenized corpus
- Log the load with chunk count

### 2. Add config setting (`src/config.py`)
Add to the `Settings` class:
```python
bm25_persist_dir: str = ".bm25"
```

### 3. Update `scripts/evaluate.py`
- Import `Path` and `get_settings` if not already imported
- Before building BM25 from ChromaDB, check if saved index exists:
  ```python
  bm25_path = Path(get_settings().bm25_persist_dir) / "bm25_index.pkl"
  ```
- If it exists and `--rebuild-bm25` flag is NOT set, load from disk
- Otherwise, build from ChromaDB and save to disk after building
- Add `--rebuild-bm25` argparse flag (store_true)

### 4. Update `app/main.py` (`_init_retriever`)
- Same pattern: try to load saved BM25 index first, fall back to building from ChromaDB
- After building from ChromaDB, save to disk
- Wrap save/load in try/except (don't crash the app on permission errors)

### Tests (`tests/test_retrieval.py`)
- Test `save_index` creates a file at the expected path (use `tmp_path` fixture)
- Test `load_index` restores a functional BM25Index
- Test `load_index` raises FileNotFoundError for missing path
- Test roundtrip: build → save → load → search produces identical results
- Test save creates parent directories if they don't exist

Run: `python -m pytest tests/test_retrieval.py -v`

### Rules
- Use pickle (BM25Okapi has complex numpy internal state)
- Add `.bm25/` to `.gitignore` if a `.gitignore` exists
- Follow project code style: type hints, Google-style docstrings, pathlib
- Commit with a descriptive message. Do NOT include Co-Authored-By.
```

---

## Prompt 7 — Final Evaluation & README Update

> **Captures**: Cumulative effect of all Phase 2 changes
> **Files to modify**: `README.md`

### Clear context, then paste:

```
## Context
I have a Financial RAG Agent project. I've just completed a series of improvements:

1. Fixed RAGAS embed_query compatibility — answer_relevancy metric now computes
2. Improved context recall — retrieval depth 5→10, financial synonym query expansion, better BM25 tokenization
3. Improved context precision — integrated cross-encoder reranker into pipeline, added query-type section boosting
4. Improved citation accuracy — component-based matching, citation metadata prepended to context strings
5. Improved faithfulness — CRAG threshold 0.6→0.7, wider context evaluation, post-generation grounding verification
6. Added BM25 persistence — save/load to avoid cold-start rebuild

Previous baseline scores:
| Metric | Score |
|---|---|
| Faithfulness | 0.60 |
| Context Precision | 0.49 |
| Context Recall | 0.35 |
| Answer Relevancy | N/A |
| Citation Accuracy | 0.54 |
| Numerical Accuracy | 0.72 |

## Task

### Step 1: Run the full test suite
Run `python -m pytest tests/ -v` and fix ANY test failures before proceeding. Do not skip this step.

### Step 2: Run the RAGAS evaluation
Run `python scripts/evaluate.py`
This evaluates all 50 test questions against the full pipeline and produces a results JSON.
This will take several minutes due to LLM API calls.

### Step 3: Update the Evaluation Results table in `README.md`
The evaluation table starts at approximately line 103 in README.md. Update it with the new scores
from the evaluation output. For each metric, update the score and add a note about what changed.

Example format:
```markdown
| Metric | Score | Notes |
|---|---|---|
| Faithfulness | **0.XX** | Improved via CRAG threshold increase and post-generation grounding verification |
| Context Precision | **0.XX** | Improved via cross-encoder reranker integration and query-type section boosting |
| Context Recall | **0.XX** | Improved via retrieval depth increase, query expansion, and BM25 tokenization |
| Answer Relevancy | **0.XX** | Now computed — fixed with LlamaIndex embedding adapter for ragas |
| Citation Accuracy | **0.XX** | Improved via component-based matching and citation metadata enrichment |
| Numerical Accuracy | **0.XX** | Custom metric: extracted numbers within 1% tolerance of ground truth |
```

Update the evaluation footnote too — remove the asterisk note about answer_relevancy being
broken since it should now work.

### Step 4: Update "Known Limitations & Future Work" in `README.md`
This section starts around line 234.

**Remove these resolved items from "Current limitations":**
- "BM25 index is rebuilt in-memory on each Streamlit startup (no persistence)" — RESOLVED
- "RAGAS answer_relevancy metric incompatible with LlamaIndex OpenAI embeddings" — RESOLVED

**Update "Evaluation-informed improvements" section:**
- Update the scores to reflect new values
- For any metric that is now ≥ 0.7, move it out of this section or note it as resolved
- Keep items for any metrics still below 0.7, but update the descriptions to reflect what was
  already done and what further improvements could help

**Keep these in "Planned improvements" (not done in this phase):**
- Full XBRL instance document parsing for broader fact coverage
- Async parallel retrieval (dense + sparse concurrently)
- Streaming LLM responses in the Streamlit chat interface
- Multi-filing temporal analysis (auto-compare across fiscal years)
- GPU-accelerated reranking for production latency targets

**Add these new items to "Planned improvements":**
- Query-specific prompt templates (different prompts for numerical vs analytical vs comparative queries)
- Chunk-level relevance feedback loop (use evaluation results to fine-tune retrieval weights)
- Multi-hop reasoning for complex comparative questions
- Evaluation caching for faster iteration (cache LLM responses for unchanged questions)

### Step 5: Commit
Stage the updated `README.md` and any evaluation result files.
Commit with message: "docs: update README with Phase 2 evaluation results and resolved limitations"
Do NOT include Co-Authored-By in the commit message.

### Rules
- Do NOT modify any source code in this prompt — only README.md
- If tests fail, fix them before running evaluation (ask me if unsure)
- Use the EXACT scores from the evaluation output — do not estimate or predict scores
- If a metric got worse, still report it honestly and add a note about investigating
```

---

## Summary

| Prompt | Focus | Key Files | Ultrathink |
|--------|-------|-----------|------------|
| 1 | RAGAS embed_query fix | `ragas_eval.py`, `test_evaluation.py` | Yes |
| 2 | Context Recall (0.35) | `hybrid.py`, `bm25_search.py`, `query_engine.py`, `financial_tools.py` | Yes |
| 3 | Context Precision (0.49) | `hybrid.py`, `evaluate.py`, `main.py` | Yes |
| 4 | Citation Accuracy (0.54) | `ragas_eval.py`, `test_evaluation.py` | Yes |
| 5 | Faithfulness (0.60) | `query_engine.py`, `test_agents.py` | Yes |
| 6 | BM25 Persistence | `bm25_search.py`, `config.py`, `evaluate.py`, `main.py` | No |
| 7 | Final Eval + README | `README.md` | No |

### Dependency Chain
```
P1 (embed_query) ─────────────────────────────────────────────────┐
P2 (recall) → P3 (precision) → P4 (citation) → P5 (faithfulness) ├→ P7 (eval + README)
P6 (BM25 persist) ────────────────────────────────────────────────┘
```

> Prompts 1 and 2 are independent of each other but both must precede Prompt 3.
> Prompt 6 is independent of all metric-improvement prompts.
> All must complete before Prompt 7.
