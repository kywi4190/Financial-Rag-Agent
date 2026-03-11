"""Chat-style Q&A interface for querying SEC filings.

Provides a conversational UI with message history, inline citations,
and retrieval metadata display.
"""

import streamlit as st

from src.agents.models import AnswerWithCitations, Citation


EXAMPLE_QUESTIONS = [
    "What was Apple's 2024 revenue?",
    "Compare debt-to-equity ratios across AAPL, MSFT, GOOGL",
    "What are the key risk factors for Apple's cloud business?",
]


def _render_citation(citation: Citation, idx: int) -> None:
    """Render a single citation as an expandable section."""
    label = f"[{idx}] {citation.ticker} {citation.year} — {citation.section}"
    with st.expander(label):
        st.caption(f"**Source:** {citation.source_document}")
        st.markdown(f"> {citation.quote_snippet}")


def _render_answer(answer: AnswerWithCitations) -> None:
    """Render an answer with citations and metadata."""
    st.markdown(answer.answer)

    # Retrieval metadata
    col1, col2 = st.columns(2)
    with col1:
        confidence_pct = answer.confidence * 100
        if answer.confidence >= 0.7:
            st.success(f"Confidence: {confidence_pct:.0f}%")
        elif answer.confidence >= 0.4:
            st.warning(f"Confidence: {confidence_pct:.0f}%")
        else:
            st.error(f"Confidence: {confidence_pct:.0f}%")
    with col2:
        st.info(f"Citations: {len(answer.citations)}")

    # Inline citations
    if answer.citations:
        st.markdown("**Sources:**")
        for i, citation in enumerate(answer.citations, 1):
            _render_citation(citation, i)


def render_chat_tab(query_engine: object | None) -> None:
    """Render the Ask Questions tab.

    Args:
        query_engine: Initialized FinancialQueryEngine, or None if unavailable.
    """
    if query_engine is None:
        st.warning(
            "No data has been ingested yet. Run "
            "`python scripts/ingest.py --ticker AAPL` to get started."
        )
        return

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Example question buttons
    st.markdown("**Try an example:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, question in zip(cols, EXAMPLE_QUESTIONS):
        with col:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

    st.divider()

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "answer_obj" in msg:
                _render_answer(msg["answer_obj"])
            else:
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about SEC filings...")

    # Handle pending question from example buttons
    pending = st.session_state.pop("pending_question", None)
    question = user_input or pending

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching filings and generating answer..."):
                # Determine ticker filter from sidebar selection
                selected_tickers = st.session_state.get("selected_tickers", [])
                ticker_filter = selected_tickers[0] if len(selected_tickers) == 1 else None

                answer = query_engine.query(question, ticker=ticker_filter)
                _render_answer(answer)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.answer,
            "answer_obj": answer,
        })
