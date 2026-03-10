# Financial Document RAG Agent

## Project Overview
SEC filing RAG system with agentic analysis. Python 3.11+, LlamaIndex, ChromaDB, Streamlit.
Portfolio project for ML engineering internships at finance AI startups.

## Build & Run Commands
- `pip install -r requirements.txt` — install dependencies
- `python -m pytest tests/ -v` — run all tests
- `streamlit run app/main.py` — launch Streamlit UI
- `python scripts/ingest.py --ticker AAPL` — ingest filings for a company
- `python scripts/evaluate.py` — run RAGAS evaluation

## Architecture
- `app/` — Streamlit frontend
- `src/ingestion/` — SEC EDGAR download + parsing (edgartools)
- `src/chunking/` — Structure-aware document chunking with metadata
- `src/retrieval/` — Hybrid search (BM25 + dense vectors + reranking)
- `src/agents/` — LlamaIndex agent definitions + tool configs
- `src/evaluation/` — RAGAS evaluation harness
- `tests/` — Pytest test suite mirroring src/ structure

## Code Style
- Type hints on ALL function signatures
- Pydantic models for ALL structured data
- Google-style docstrings on public functions
- Use pathlib, not os.path
- Prefer composition over inheritance
- Use async where appropriate for I/O operations

## Rules
- NEVER hardcode API keys — use .env via python-dotenv
- ALWAYS preserve financial tables as markdown — never split a table across chunks
- EVERY generated claim must include a source citation (document, section, page)
- Run pytest after every feature implementation
- Commit after each completed feature with descriptive message
- Log all LLM calls with input/output for debugging
