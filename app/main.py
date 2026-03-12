"""Streamlit entry point for the Financial RAG Agent.

Provides an interactive UI for querying SEC filings, viewing retrieved
context, and generating investment memos.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# Bridge Streamlit Cloud secrets into environment variables so that
# pydantic-settings (src/config.py) can pick them up transparently.
if hasattr(st, "secrets"):
    for key in ("OPENAI_API_KEY", "SEC_EDGAR_IDENTITY"):
        if key in st.secrets and key not in os.environ:
            os.environ[key] = st.secrets[key]

from app.components.chat import render_chat_tab
from app.components.memo import render_memo_tab
from app.components.metrics import render_metrics_tab

logger = logging.getLogger(__name__)


def _check_api_keys() -> bool:
    """Return True if required API keys are configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


@st.cache_resource
def _init_vector_store() -> "ChromaStore | None":
    """Initialize and cache the ChromaDB vector store."""
    try:
        from src.retrieval.vector_store import ChromaStore

        store = ChromaStore()
        stats = store.get_collection_stats()
        if stats.get("count", 0) == 0:
            return None
        return store
    except Exception:
        logger.exception("Failed to initialize vector store")
        return None


@st.cache_resource
def _init_retriever(_vector_store: "ChromaStore") -> "HybridRetriever | None":
    """Initialize the hybrid retriever from an existing vector store.

    Args:
        _vector_store: Initialized ChromaStore with indexed documents.
    """
    try:
        from src.retrieval.bm25_search import BM25Index
        from src.retrieval.hybrid import HybridRetriever
        from src.retrieval.models import RetrievalConfig

        # Build BM25 index from ChromaDB documents
        bm25 = BM25Index()
        all_docs = _vector_store._collection.get(include=["documents", "metadatas"])
        if all_docs["documents"]:
            from src.chunking.models import ChunkMetadata, DocumentChunk

            chunks = []
            for i, doc in enumerate(all_docs["documents"]):
                meta = all_docs["metadatas"][i]
                chunks.append(DocumentChunk(
                    chunk_id=all_docs["ids"][i],
                    content=doc,
                    metadata=ChunkMetadata(**meta),
                ))
            bm25.build_index(chunks)

        config = RetrievalConfig()

        reranker = None
        try:
            from src.retrieval.reranker import Reranker

            reranker = Reranker()
        except Exception:
            logger.warning("Reranker unavailable, continuing without cross-encoder reranking")

        return HybridRetriever(_vector_store, bm25, config, reranker=reranker)
    except Exception:
        logger.exception("Failed to initialize retriever")
        return None


@st.cache_resource
def _init_query_engine(_retriever: "HybridRetriever") -> "FinancialQueryEngine | None":
    """Initialize the query engine.

    Args:
        _retriever: Initialized HybridRetriever.
    """
    try:
        from src.agents.query_engine import FinancialQueryEngine

        return FinancialQueryEngine(retriever=_retriever)
    except Exception:
        logger.exception("Failed to initialize query engine")
        return None


@st.cache_resource
def _init_memo_generator(_retriever: "HybridRetriever") -> "MemoGenerator | None":
    """Initialize the memo generator.

    Args:
        _retriever: Initialized HybridRetriever.
    """
    try:
        from src.agents.memo_generator import MemoGenerator

        return MemoGenerator(retriever=_retriever)
    except Exception:
        logger.exception("Failed to initialize memo generator")
        return None


def _render_sidebar(collection_stats: dict | None) -> None:
    """Render the sidebar with filters and model settings."""
    with st.sidebar:
        st.header("Filters")

        # Company selector
        if collection_stats and collection_stats.get("tickers"):
            tickers = collection_stats["tickers"]
            selected = []
            st.subheader("Companies")
            for ticker in tickers:
                if st.checkbox(ticker, value=True, key=f"ticker_{ticker}"):
                    selected.append(ticker)
            st.session_state.selected_tickers = selected

            # Year filter
            if collection_stats.get("years"):
                years = sorted(collection_stats["years"], reverse=True)
                st.session_state.available_years = years
                st.subheader("Filing Years")
                selected_years = st.multiselect(
                    "Filter by year",
                    years,
                    default=years,
                    key="year_filter",
                )
                st.session_state.selected_years = selected_years
        else:
            st.info("No companies ingested yet.")
            st.session_state.selected_tickers = []
            st.session_state.available_years = []

        st.divider()

        # Model settings
        st.subheader("Model Settings")
        st.text_input("LLM Model", value="gpt-4o-mini", key="llm_model", disabled=True)
        st.text_input(
            "Embedding Model",
            value="text-embedding-3-small",
            key="embed_model",
            disabled=True,
        )


def main() -> None:
    """Launch the Streamlit application."""
    st.set_page_config(
        page_title="Financial RAG Agent",
        page_icon="\U0001f4ca",
        layout="wide",
    )

    st.title("Financial RAG Agent")
    st.caption("SEC filing analysis with agentic RAG")

    # Gate on required API keys before initializing any backend
    if not _check_api_keys():
        st.error("**Missing API keys.** The app cannot start without them.")
        st.markdown(
            "**Local development:**\n"
            "```bash\n"
            "cp .env.example .env\n"
            "# edit .env and fill in your OPENAI_API_KEY\n"
            "```\n\n"
            "**Streamlit Community Cloud:**\n"
            "1. Open your app dashboard on [share.streamlit.io](https://share.streamlit.io)\n"
            "2. Go to **Settings > Secrets**\n"
            "3. Paste the contents of `.streamlit/secrets.toml.example` and fill in real values\n"
        )
        st.stop()

    # Initialize backend components
    vector_store = _init_vector_store()
    collection_stats = None
    retriever = None
    query_engine = None
    memo_generator = None

    if vector_store is not None:
        collection_stats = vector_store.get_collection_stats()
        retriever = _init_retriever(vector_store)
        if retriever is not None:
            query_engine = _init_query_engine(retriever)
            memo_generator = _init_memo_generator(retriever)

    # Sidebar
    _render_sidebar(collection_stats)

    # Main tabs
    tab_qa, tab_memo, tab_metrics = st.tabs([
        "Ask Questions",
        "Investment Memo",
        "System Metrics",
    ])

    with tab_qa:
        render_chat_tab(query_engine)

    with tab_memo:
        available_tickers = (
            collection_stats.get("tickers", []) if collection_stats else []
        )
        render_memo_tab(memo_generator, available_tickers)

    with tab_metrics:
        render_metrics_tab(collection_stats)

    # Footer
    st.divider()
    st.caption("Powered by LlamaIndex + ChromaDB + GPT-4o-mini")


if __name__ == "__main__":
    main()
