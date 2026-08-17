from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from eda.lib import load_json


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _folder_table(scans: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in scans:
        vs = s.get("value_stats") or {}
        cad = s.get("cadence") or {}
        rows.append(
            {
                "file": s.get("file_name"),
                "rows": s.get("rows"),
                "time_min": s.get("timestamp_min"),
                "time_max": s.get("timestamp_max"),
                "cadence_median_s": _safe_float(cad.get("median_seconds")),
                "value_mean": _safe_float(vs.get("mean")),
                "value_p95": _safe_float(vs.get("p95")),
                "value_max": _safe_float(vs.get("max")),
            }
        )
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="CrossLayerAI-RAN EDA", layout="wide")
    st.title("CrossLayerAI-RAN Dataset EDA")

    st.sidebar.header("Inputs")
    summary_path = Path(st.sidebar.text_input("summary.json path", value="artifacts/eda/summary.json"))
    if not summary_path.exists():
        st.warning(f"Missing {summary_path}. Build it first: `python -m eda.build_eda`")
        st.stop()

    summary = load_json(summary_path)
    st.caption(f"Generated at {summary.get('generated_at')} | dataset: `{summary.get('dataset_dir')}`")

    folders = list(summary.get("folders", {}).keys())
    if not folders:
        st.error("No folders found in summary.")
        st.stop()

    folder = st.sidebar.selectbox("Folder", folders, index=0)
    scans = summary["folders"][folder]["scans"]

    st.subheader(f"{folder} ({summary['folders'][folder]['files_scanned']} files scanned)")
    df = _folder_table(scans)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.sort_values("rows", ascending=False), use_container_width=True, height=420)
    with c2:
        st.markdown("**Scan configuration**")
        st.json(summary.get("config", {}))

    st.divider()
    st.subheader("Plots (based on scanned files)")
    p1, p2 = st.columns(2)
    with p1:
        fig = px.histogram(df, x="cadence_median_s", nbins=30, title="Cadence median (seconds)")
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        fig = px.histogram(df, x="value_mean", nbins=30, title="Value mean (sample)")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Categorical label distributions (top-k per file, aggregated)")
    # Aggregate top labels across scanned files
    label_counts: Dict[str, Dict[str, int]] = {}
    for s in scans:
        cat = s.get("categorical_top") or {}
        for col, entries in cat.items():
            d = label_counts.setdefault(col, {})
            for e in entries:
                v = str(e.get("value"))
                d[v] = d.get(v, 0) + int(e.get("count") or 0)

    if not label_counts:
        st.info("No categorical columns detected in scanned files.")
    else:
        col = st.selectbox("Categorical column", sorted(label_counts.keys()))
        items = sorted(label_counts[col].items(), key=lambda kv: kv[1], reverse=True)[:40]
        cdf = pd.DataFrame(items, columns=["label", "count"])
        fig = px.bar(cdf, x="label", y="count", title=f"{col}: top labels (aggregated)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

