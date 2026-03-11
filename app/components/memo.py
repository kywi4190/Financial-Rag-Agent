"""Investment memo generation tab.

Provides ticker selection, progress tracking, and formatted memo
rendering with expandable citations and markdown download.
"""

import streamlit as st

from src.agents.models import InvestmentMemo, MemoSection


def _render_memo_section(section: MemoSection) -> None:
    """Render a single memo section with expandable citations."""
    st.markdown(f"### {section.title}")
    st.markdown(section.content)

    if section.citations:
        with st.expander(f"Sources ({len(section.citations)})"):
            for c in section.citations:
                st.markdown(
                    f"- **{c.source_document}**, {c.section} — "
                    f"*\"{c.quote_snippet}\"*"
                )


def _render_memo(memo: InvestmentMemo) -> None:
    """Render the full investment memo."""
    st.markdown(f"## Investment Memo: {memo.ticker} — {memo.company_name}")
    st.caption(f"Generated: {memo.date_generated.strftime('%Y-%m-%d %H:%M UTC')}")

    sections = [
        memo.executive_summary,
        memo.company_overview,
        memo.financial_highlights,
        memo.risk_factors,
        memo.mda_synthesis,
        memo.bull_bear_case,
    ]

    for section in sections:
        _render_memo_section(section)
        st.divider()


def render_memo_tab(memo_generator: object | None, available_tickers: list[str]) -> None:
    """Render the Investment Memo tab.

    Args:
        memo_generator: Initialized MemoGenerator, or None if unavailable.
        available_tickers: List of ingested ticker symbols.
    """
    if memo_generator is None or not available_tickers:
        st.warning(
            "No data has been ingested yet. Run "
            "`python scripts/ingest.py --ticker AAPL` to get started."
        )
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ticker = st.selectbox("Select company", available_tickers, key="memo_ticker")
    with col2:
        available_years = st.session_state.get("available_years", [])
        year_options = ["Latest"] + [str(y) for y in sorted(available_years, reverse=True)]
        year_selection = st.selectbox("Fiscal year", year_options, key="memo_year")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("Generate Investment Memo", type="primary")

    if generate:
        year = int(year_selection) if year_selection != "Latest" else None

        progress = st.progress(0, text="Starting memo generation...")
        status = st.empty()

        stages = [
            (0.10, "Resolving company name..."),
            (0.25, "Extracting financial data (XBRL)..."),
            (0.45, "Analyzing company overview..."),
            (0.60, "Extracting risk factors..."),
            (0.75, "Synthesizing MD&A..."),
            (0.85, "Generating executive summary..."),
            (0.95, "Building bull & bear cases..."),
        ]

        # Show progressive updates while generating
        import threading
        import time

        progress_done = threading.Event()

        def _update_progress() -> None:
            for pct, msg in stages:
                if progress_done.is_set():
                    break
                progress.progress(pct, text=msg)
                status.caption(msg)
                time.sleep(2.5)

        thread = threading.Thread(target=_update_progress, daemon=True)
        thread.start()

        try:
            memo = memo_generator.generate_memo(ticker, year=year)
            progress_done.set()
            thread.join(timeout=1)
            progress.progress(1.0, text="Memo complete!")
            status.empty()

            st.session_state.last_memo = memo
        except Exception as e:
            progress_done.set()
            thread.join(timeout=1)
            progress.empty()
            status.empty()
            st.error(f"Memo generation failed: {e}")
            return

    # Display memo if available
    memo = st.session_state.get("last_memo")
    if memo is not None:
        st.divider()
        _render_memo(memo)

        # Download button
        markdown_content = memo.to_markdown()
        st.download_button(
            label="Download as Markdown",
            data=markdown_content,
            file_name=f"memo_{memo.ticker}_{memo.date_generated.strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )
