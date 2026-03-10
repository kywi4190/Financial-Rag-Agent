"""Streamlit entry point for the Financial RAG Agent.

Provides an interactive UI for querying SEC filings, viewing retrieved
context, and generating investment memos.
"""

import streamlit as st


def main() -> None:
    """Launch the Streamlit application."""
    st.set_page_config(
        page_title="Financial RAG Agent",
        layout="wide",
    )
    st.title("Financial RAG Agent")
    st.info("Application under construction.")


if __name__ == "__main__":
    main()
