"""System metrics and RAGAS evaluation dashboard.

Displays evaluation scores, per-category breakdowns, score distribution
charts, and ingested data summaries.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _load_latest_eval_report(eval_dir: str = "data/eval") -> dict | None:
    """Load the most recent evaluation report JSON.

    Args:
        eval_dir: Directory containing evaluation result files.

    Returns:
        Parsed report dict, or None if no reports found.
    """
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        return None

    result_files = sorted(eval_path.glob("results_*.json"), reverse=True)
    if not result_files:
        return None

    with open(result_files[0]) as f:
        return json.load(f)


def _render_overall_scores(scores: dict[str, float]) -> None:
    """Render overall metric scores as metric cards."""
    st.subheader("Overall Scores")

    metric_labels = {
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "answer_relevancy": "Answer Relevancy",
        "citation_accuracy": "Citation Accuracy",
    }

    cols = st.columns(len(metric_labels))
    for col, (key, label) in zip(cols, metric_labels.items()):
        value = scores.get(key, 0.0)
        with col:
            st.metric(label, f"{value:.2f}")


def _render_category_breakdown(per_category: dict[str, dict[str, float]]) -> None:
    """Render per-category score breakdown as a bar chart."""
    st.subheader("Scores by Question Category")

    if not per_category:
        st.info("No per-category data available.")
        return

    # Build DataFrame for chart: rows = metrics, columns = categories
    all_metrics: set[str] = set()
    for scores in per_category.values():
        all_metrics.update(scores.keys())

    chart_data: dict[str, list[float]] = {}
    categories = sorted(per_category.keys())
    for metric in sorted(all_metrics):
        chart_data[metric] = [
            per_category[cat].get(metric, 0.0) for cat in categories
        ]

    df = pd.DataFrame(chart_data, index=categories)
    st.bar_chart(df)

    # Also show as a table for detail
    with st.expander("Detailed scores table"):
        st.dataframe(df.T.style.format("{:.3f}"), use_container_width=True)


def _render_ingested_summary(collection_stats: dict) -> None:
    """Render summary of ingested data."""
    st.subheader("Ingested Data Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        tickers = collection_stats.get("tickers", [])
        st.metric("Companies", len(tickers))
        if tickers:
            st.caption(", ".join(tickers))
    with col2:
        years = collection_stats.get("years", [])
        st.metric("Filing Years", len(years))
        if years:
            st.caption(", ".join(str(y) for y in years))
    with col3:
        count = collection_stats.get("count", 0)
        st.metric("Total Chunks / Embeddings", f"{count:,}")


def render_metrics_tab(collection_stats: dict | None) -> None:
    """Render the System Metrics tab.

    Args:
        collection_stats: Dict from ChromaStore.get_collection_stats(), or None.
    """
    # Ingested data summary
    if collection_stats and collection_stats.get("count", 0) > 0:
        _render_ingested_summary(collection_stats)
    else:
        st.info(
            "No data ingested. Run `python scripts/ingest.py --ticker AAPL` "
            "to index filings, then `python scripts/evaluate.py` to generate scores."
        )

    st.divider()

    # RAGAS evaluation scores
    report = _load_latest_eval_report()
    if report is None:
        st.info(
            "No evaluation results found. Run `python scripts/evaluate.py` "
            "to generate RAGAS scores."
        )
        return

    _render_overall_scores(report.get("overall_scores", {}))
    st.divider()
    _render_category_breakdown(report.get("per_category_scores", {}))
