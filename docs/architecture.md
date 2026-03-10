# Architecture Decisions Log

This document tracks key architectural decisions for the Financial RAG Agent project.

## ADR-001: Hybrid Retrieval (Vector + BM25 with RRF)

**Context:** Financial documents contain both semantic concepts and precise terminology (ticker symbols, GAAP terms, numeric values) that pure vector search can miss.

**Decision:** Combine dense vector search (ChromaDB + OpenAI embeddings) with BM25 sparse search, fused via Reciprocal Rank Fusion.

**Status:** Proposed

---

## ADR-002: Cross-Encoder Reranking

**Context:** Initial retrieval returns a broad set of candidates; precision matters for financial Q&A.

**Decision:** Apply a cross-encoder reranker (sentence-transformers) as a second-stage filter to improve top-k precision.

**Status:** Proposed

---

## ADR-003: Structure-Aware Chunking

**Context:** Financial filings have well-defined sections (Item 1, Item 7, etc.) and tables that should not be split mid-structure.

**Decision:** Custom chunker that respects section boundaries and treats tables as atomic units.

**Status:** Proposed

---

## ADR-004: RAGAS Evaluation Framework

**Context:** Need quantitative measurement of RAG pipeline quality across faithfulness, relevancy, precision, and recall.

**Decision:** Use RAGAS framework with synthetic test question generation for reproducible evaluation.

**Status:** Proposed
