import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JLab Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths — check D:/ (original training machine) then fall back to relative ──
_d_drive = Path("D:/CrossLayer-RAN-Samsung-Prism--master/artifacts/models")
_local   = Path(__file__).parent / "artifacts" / "models"
MODELS_DIR  = _d_drive if _d_drive.exists() else _local
SCORES_DIR  = MODELS_DIR  # per-cluster anomaly score parquets saved alongside models

# ── Load static JSON results ──────────────────────────────────────────────────
@st.cache_data
def load_results():
    """Load best available results. v7 = all 3 modalities + Adaptive + LOF."""
    for ver in ("v7", "v6", "v5", "v4", "v3", ""):
        suffix = f"_{ver}" if ver else ""
        r_path = MODELS_DIR / f"multimodel_results{suffix}.json"
        m_path = MODELS_DIR / f"evaluation_report_metrics{suffix}.json"
        if r_path.exists() and m_path.exists():
            with open(r_path) as f:
                results = json.load(f)
            with open(m_path) as f:
                metrics = json.load(f)
            return results, metrics, ver or "original"
    st.error("No result JSON files found in " + str(MODELS_DIR))
    st.stop()


@st.cache_data
def get_modality_data(results, metrics, modality):
    """Extract farm-level results and metrics for a given modality."""
    # v6 structure: results["by_modality"][modality][farm]
    # v4 structure: results[farm] (flat, memory+slurm only)
    if "by_modality" in results:
        r = results["by_modality"].get(modality, {})
    else:
        r = results  # legacy flat format

    if "by_modality" in metrics:
        m = metrics["by_modality"].get(modality, {})
    else:
        m = metrics

    return r, m

@st.cache_data
def load_thresholds():
    path = MODELS_DIR / "thresholds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

# ── Load ML models for a given cluster ───────────────────────────────────────
@st.cache_resource
def load_models(cluster, modality="memory_slurm"):
    prefix = {"memory_slurm": "", "cpu": "cpu_", "disk": "disk_"}[modality]
    ae  = joblib.load(MODELS_DIR / f"{prefix}autoencoder_{cluster}.joblib")
    sc  = joblib.load(MODELS_DIR / f"{prefix}scaler_{cluster}.joblib")
    iso = joblib.load(MODELS_DIR / f"{prefix}isoforest_{cluster}.joblib")
    return ae, sc, iso

# ── Load per-row anomaly score parquet for a modality ─────────────────────────
@st.cache_data
def load_score_ts(cluster, modality="memory_slurm"):
    # v6 unified score files per modality
    candidates = {
        "memory_slurm": [f"anomaly_scores_{cluster}.parquet"],
        "cpu":          ["cpu_anomaly_scores.parquet"],
        "disk":         ["disk_anomaly_scores.parquet"],
    }
    # Also expose LOF score files when available
    lof_candidates = {
        "cpu":  "cpu_lof_scores.parquet",
        "disk": "disk_lof_scores.parquet",
    }
    if modality in lof_candidates:
        candidates[modality].append(lof_candidates[modality])
    for fname in candidates.get(modality, []):
        path = SCORES_DIR / fname
        if path.exists():
            df = pd.read_parquet(path)
            # Filter to cluster if multi-farm file
            if "hw_config" in df.columns:
                df = df[df["hw_config"] == cluster]
            if "split" in df.columns:
                df = df[df["split"] == "test"]
            return df if not df.empty else None
    return None

results, metrics, result_version = load_results()
thresholds_data = load_thresholds()

# Detect v6 unified format vs legacy flat format
IS_V6 = "by_modality" in results
AVAIL_MODALITIES = list(results["by_modality"].keys()) if IS_V6 else ["memory_slurm"]
MODALITY_LABELS  = {"memory_slurm": "🧠 Memory + SLURM", "cpu": "⚡ CPU", "disk": "💾 Disk"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Samsung_Logo.svg/1920px-Samsung_Logo.svg.png",
    width=140
)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Results Dashboard", "📈 Time-Series Scores", "🚨 Live Anomaly Playground"]
)

# Modality selector (only shown for v6)
if IS_V6:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Modality")
    selected_modality = st.sidebar.selectbox(
        "Data source",
        options=AVAIL_MODALITIES,
        format_func=lambda x: MODALITY_LABELS.get(x, x),
        key="modality_select"
    )
else:
    selected_modality = "memory_slurm"

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline")
for i, step in enumerate(
    ["Structural Cleaning", "Chronological Split", "IQR Normalization",
     "Feature Engineering", "Multimodal Join", "Multimodel Training"], 1
):
    st.sidebar.markdown(f"**{i}.** {step}")

if result_version != "original":
    st.sidebar.success(f"Results: v{result_version} · {len(AVAIL_MODALITIES)} modalities")
    if result_version == "v7":
        st.sidebar.caption("✨ v7: +Adaptive Threshold +LOF (CPU & Disk)")
else:
    st.sidebar.info("Results: original")

if thresholds_data:
    st.sidebar.success("✓ Real thresholds loaded")
else:
    st.sidebar.warning("⚠ Thresholds not found")

# ── Resolve active farm results for selected modality ─────────────────────────
farm_results, farm_metrics = get_modality_data(results, metrics, selected_modality)
clusters = [c for c in farm_results.keys() if isinstance(farm_results[c], dict)
            and "test_rows" in farm_results[c]]

# ── Global KPIs ───────────────────────────────────────────────────────────────
total_train    = sum(farm_results[c].get("test_rows", 0)    for c in clusters)
total_distress = sum(farm_results[c].get("test_distress_rows", 0) for c in clusters)
total_caught   = sum(
    farm_results[c].get("Autoencoder", {}).get("caught_distress", 0)
    if "Autoencoder" in farm_results[c]
    else farm_results[c].get("models", {}).get("Autoencoder", {}).get("caught_distress", 0)
    for c in clusters
)

def safe_div(n, d):
    return n / d if d > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Results Dashboard":
    st.title("📡 JLab Cluster Anomaly Detection")
    st.caption("Samsung PRISM · Unsupervised ML Pipeline · 180GB+ Dataset · CPU + Memory + Disk + SLURM")

    # Modality banner
    if IS_V6:
        mod_label = MODALITY_LABELS.get(selected_modality, selected_modality)
        v_label = f"v{result_version}" if result_version != "original" else "original"
        st.info(f"Showing: **{mod_label}** — switch modality in the sidebar.  "
                f"Metrics: `evaluation_report_metrics_{v_label}.json`  "
                f"{'· 🆕 Adaptive Thresholding + LOF for CPU & Disk' if result_version == 'v7' else ''}")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Rows", f"{total_train/1e6:.2f}M")
    c2.metric("Hardware Farms", str(len(clusters)))
    c3.metric("Distress Events (May 23)", f"{total_distress:,}")
    c4.metric("Caught by Autoencoder", f"{total_caught:,}")
    c5.metric(
        "AE Overall Recall",
        f"{total_caught/total_distress*100:.1f}%" if total_distress > 0 else "N/A"
    )
    st.markdown("---")

    # ── Recall by model table ─────────────────────────────────────────────────
    st.subheader("Evaluation Metrics by Model Across All Farms")
    table_rows = []
    # Get model keys from first farm
    # Normalize: cpu stores model stats at farm top-level (empty 'models' dict);
    # disk stores at top-level (no 'models' key); memory_slurm uses 'models' sub-key.
    _NON_MODEL_KEYS = {"status", "modality", "test_rows", "test_distress_rows", "train_rows", "models"}
    def _get_model_data(farm_dict):
        """Return the dict containing per-model stats, normalised across all formats."""
        sub = farm_dict.get("models", {})
        if sub:  # non-empty 'models' sub-key (memory_slurm)
            return sub
        # Empty or absent 'models' key → model entries are at top-level (cpu / disk)
        return {k: v for k, v in farm_dict.items()
                if k not in _NON_MODEL_KEYS and isinstance(v, dict)}

    first_farm = clusters[0] if clusters else None
    if first_farm:
        model_keys = list(_get_model_data(farm_results[first_farm]).keys())
    else:
        model_keys = []

    for c in clusters:
        nd = farm_results[c].get("test_distress_rows", 0)
        mdata = _get_model_data(farm_results[c])
        for model in model_keys:
            if model not in mdata: continue
            caught  = mdata[model].get("caught_distress", 0)
            flagged = mdata[model].get("test_anomalies", 0)
            recall    = f"{caught/nd*100:.1f}%" if nd > 0 else "N/A"
            precision = f"{safe_div(caught, flagged)*100:.1f}%" if flagged > 0 else "N/A"
            fm = farm_metrics.get(c, {}).get(model, {})
            f1_val  = fm.get("f1_score") or 0
            fpr_raw = fm.get("false_positive_rate")
            rr_val  = fm.get("robust_recall") or 0
            table_rows.append({
                "Farm": c, "Model": model,
                "Distress Rows": nd, "Caught": caught,
                "Recall": recall, "Precision": precision,
                "F1": f"{f1_val*100:.2f}%",
                "FPR": f"{fpr_raw*100:.2f}%" if fpr_raw is not None else "N/A",
                "Robust Recall": f"{rr_val*100:.1f}%",
            })
    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, hide_index=True)

    # ── Bar chart: Recall per farm (all models) ───────────────────────────────
    st.subheader("Recall per Farm — All Models")
    recall_rows = []
    for c in clusters:
        nd   = farm_results[c].get("test_distress_rows", 0)
        if nd == 0: continue
        mdata = _get_model_data(farm_results[c])
        for model in model_keys:
            if model not in mdata: continue
            caught = mdata[model].get("caught_distress", 0)
            recall_rows.append({"Farm": c, "Model": model, "Recall (%)": safe_div(caught, nd)*100})
    if recall_rows:
        fig_recall = px.bar(
            pd.DataFrame(recall_rows), x="Farm", y="Recall (%)",
            color="Model", barmode="group",
            color_discrete_map={
                "Autoencoder":    "#1f77b4",
                "IsolationForest":"#ff7f0e",
                "LOF":            "#2ca02c",
                "Ensemble":       "#d62728",
                "AE_Adaptive":    "#9467bd",
                "Ens_Adaptive":   "#8c564b",
            },
        )
        st.plotly_chart(fig_recall, use_container_width=True)

    # ── Precision vs Recall scatter (Plotly) ──────────────────────────────────
    st.subheader("Precision vs Recall — All Models & Farms")
    st.caption("Ideal model = top-right corner. Size = F1 score. FPR shown on hover.")
    pr_rows = []
    for c in clusters:
        nd = farm_results[c].get("test_distress_rows", 0)
        if nd == 0: continue
        for model in model_keys:
            fm = farm_metrics.get(c, {}).get(model, {})
            if not fm: continue
            pr_rows.append({
                "Farm": c, "Model": model,
                "Recall (%)": (fm.get("recall") or 0) * 100,
                "Precision (%)": (fm.get("precision") or 0) * 100,
                "F1 (%)": (fm.get("f1_score") or 0) * 100,
                "FPR (%)": (fm.get("false_positive_rate") or 0) * 100,
                "Robust Recall (%)": (fm.get("robust_recall") or 0) * 100,
                "Latency ms/row": fm.get("latency_ms_per_row") or 0,
            })
    if pr_rows:
        pr_df = pd.DataFrame(pr_rows)
        fig_pr = px.scatter(
            pr_df, x="Recall (%)", y="Precision (%)",
            color="Model", symbol="Farm",
            size="F1 (%)", size_max=30,
            hover_data=["Farm", "Model", "F1 (%)", "FPR (%)", "Robust Recall (%)", "Latency ms/row"],
            color_discrete_map={
                "Autoencoder":    "#1f77b4",
                "IsolationForest":"#ff7f0e",
                "LOF":            "#2ca02c",
                "Ensemble":       "#d62728",
                "AE_Adaptive":    "#9467bd",
                "Ens_Adaptive":   "#8c564b",
            },
        )
        fig_pr.update_layout(
            xaxis=dict(range=[0, 105], title="Recall (%)"),
            yaxis=dict(range=[0, 105], title="Precision (%)"),
            height=450,
        )
        fig_pr.add_shape(type="line", x0=0, y0=0, x1=105, y1=105,
                         line=dict(color="gray", dash="dot"))
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Cross-modality comparison (v6 only) ───────────────────────────────────
    if IS_V6 and len(AVAIL_MODALITIES) > 1:
        st.markdown("---")
        st.subheader("Cross-Modality Recall Comparison (May 23)")
        st.caption("Which modality detects distress best per farm?")
        cross_rows = []
        # Include adaptive + LOF models when present (v7)
        cross_models = ["Autoencoder", "IsolationForest", "Ensemble",
                        "AE_Adaptive", "Ens_Adaptive", "LOF"]
        for mod in AVAIL_MODALITIES:
            fr, fm = get_modality_data(results, metrics, mod)
            for c in sorted(fr.keys()):
                if not isinstance(fr[c], dict): continue
                nd = fr[c].get("test_distress_rows", 0)
                if nd == 0: continue
                for mdl in cross_models:
                    mfm = fm.get(c, {}).get(mdl, {})
                    if mfm:
                        cross_rows.append({
                            "Modality": MODALITY_LABELS.get(mod, mod),
                            "Farm": c, "Model": mdl,
                            "Recall (%)": mfm.get("recall", 0) * 100,
                        })
        if cross_rows:
            cross_df = pd.DataFrame(cross_rows)
            fig_cross = px.bar(
                cross_df, x="Farm", y="Recall (%)", color="Modality",
                facet_col="Model", barmode="group",
                height=400,
                color_discrete_map={
                    MODALITY_LABELS["memory_slurm"]: "#636efa",
                    MODALITY_LABELS["cpu"]: "#ef553b",
                    MODALITY_LABELS["disk"]: "#00cc96",
                }
            )
            st.plotly_chart(fig_cross, use_container_width=True)

    # ── Latency & Robustness ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inference Latency (ms/row)")
        lat_rows = []
        for c in clusters:
            for model in model_keys:
                fm = farm_metrics.get(c, {}).get(model, {})
                if fm:
                    lat_rows.append({"Farm": c, "Model": model,
                                     "Latency": fm.get("latency_ms_per_row") or 0})
        if lat_rows:
            lat_df = pd.DataFrame(lat_rows)
            fig_lat = px.bar(lat_df, x="Farm", y="Latency", color="Model",
                             barmode="group", log_y=True,
                             labels={"Latency": "ms/row (log scale)"})
            st.plotly_chart(fig_lat, use_container_width=True)
    with col2:
        st.subheader("Robust Recall (Top-10% Scores)")
        rob_rows = []
        for c in clusters:
            if farm_results[c].get("test_distress_rows", 0) == 0: continue
            for model in model_keys:
                fm = farm_metrics.get(c, {}).get(model, {})
                rr = fm.get("robust_recall")
                if rr is not None:
                    rob_rows.append({"Farm": c, "Model": model, "Robust Recall (%)": rr * 100})
        if rob_rows:
            rob_df = pd.DataFrame(rob_rows)
            fig_rob = px.bar(rob_df, x="Farm", y="Robust Recall (%)",
                             color="Model", barmode="group")
            st.plotly_chart(fig_rob, use_container_width=True)

    # ── Per-farm deep dive ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Deep Dive by Farm")
    selected = st.selectbox("Select Farm", clusters)
    r   = farm_results[selected]
    nd  = r.get("test_distress_rows", 0)
    m_sel = farm_metrics.get(selected, {})
    cols  = st.columns(min(len(model_keys), 4))
    for col, model in zip(cols, model_keys):
        mm = m_sel.get(model, {})
        with col:
            st.markdown(f"**{model}**")
            st.metric("Recall",       f"{mm.get('recall',0)*100:.1f}%" if nd > 0 else "N/A")
            st.metric("Precision",    f"{mm.get('precision',0)*100:.2f}%")
            st.metric("F1",           f"{(mm.get('f1_score') or 0)*100:.2f}%")
            st.metric("FPR",          f"{(mm.get('false_positive_rate') or 0)*100:.2f}%")
            rr = mm.get("robust_recall")
            st.metric("Robust Recall", f"{rr*100:.1f}%" if rr is not None else "N/A")
            st.metric("Latency",      f"{mm.get('latency_ms_per_row',0):.4f} ms")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: TIME-SERIES SCORES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Time-Series Scores":
    st.title("📈 Anomaly Score Over Time")
    st.caption(
        "Reconstruction error (Autoencoder) per node over the JLab event window "
        "(May 19–23 2023). A spike on May 23 indicates the anomaly."
    )

    col_ts1, col_ts2 = st.columns([3, 1])
    with col_ts1:
        cluster_ts = st.selectbox("Select Farm", clusters, key="ts_cluster")
    with col_ts2:
        mod_ts = st.selectbox("Modality", AVAIL_MODALITIES,
                               format_func=lambda x: MODALITY_LABELS.get(x, x),
                               key="ts_modality") if IS_V6 else "memory_slurm"

    score_df = load_score_ts(cluster_ts, mod_ts)

    if score_df is None:
        st.warning(
            f"No per-row score file found for **{cluster_ts}** / **{mod_ts}**.\n\n"
            "Score parquets: `cpu_anomaly_scores.parquet`, `disk_anomaly_scores.parquet`."
        )
    else:
        score_df["timestamp"] = pd.to_datetime(score_df["timestamp"])
        score_df = score_df.sort_values("timestamp")

        # Hourly mean AE score across all nodes
        score_df["hour"] = score_df["timestamp"].dt.floor("h")
        ae_col = "ae_score" if "ae_score" in score_df.columns else score_df.select_dtypes("number").columns[0]
        hourly = score_df.groupby("hour")[ae_col].mean().reset_index()
        hourly.columns = ["Hour", "Mean AE Reconstruction Error"]

        thr_val  = thresholds_data.get(cluster_ts, {}).get("threshold")
        # For CPU/disk use AE threshold from training summary
        if thr_val is None and mod_ts in ("cpu", "disk"):
            try:
                ts_path = MODELS_DIR / f"{mod_ts}_training_summary.json"
                with open(ts_path) as f:
                    ts_data = json.load(f)
                thr_val = ts_data.get("models",{}).get("per_farm",{}).get(cluster_ts,{}).get("ae_thr")
            except Exception:
                pass

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["Mean AE Reconstruction Error"],
            mode="lines", name="Mean AE Error", line=dict(color="#1f77b4", width=2)
        ))
        if thr_val:
            fig_ts.add_hline(
                y=thr_val, line_dash="dash", line_color="red",
                annotation_text=f"Anomaly Threshold ({thr_val:.4f})",
                annotation_position="top left"
            )
        may23_start = pd.Timestamp("2023-05-23 00:00")
        hour_max    = pd.Timestamp(hourly["Hour"].max())
        if may23_start <= hour_max:
            fig_ts.add_vrect(
                x0=may23_start, x1=hour_max,
                fillcolor="red", opacity=0.08, line_width=0,
                annotation_text="Anomalous Event (May 23)",
                annotation_position="top left"
            )
        fig_ts.update_layout(
            xaxis_title="Time", yaxis_title="AE Reconstruction Error",
            height=400, hovermode="x unified"
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Node-level heatmap (top 30 nodes by max AE score)
        st.subheader("Node-Level Anomaly Heatmap (May 23 only)")
        may23 = score_df[score_df["timestamp"].dt.date == pd.Timestamp("2023-05-23").date()]
        node_col = "node" if "node" in may23.columns else ("instance" if "instance" in may23.columns else None)
        if not may23.empty and node_col:
            may23 = may23.copy()
            may23["hour"] = may23["timestamp"].dt.floor("h")
            pivot = may23.pivot_table(
                index=node_col, columns="hour", values=ae_col,
                aggfunc="mean"
            )
            top_nodes = pivot.max(axis=1).nlargest(30).index
            pivot = pivot.loc[top_nodes]

            # Log-scale the values so a single huge outlier doesn't flatten the rest
            pivot_log = np.log1p(pivot.fillna(0))

            # Tick labels in original units (not log) for the colorbar
            raw_vals     = pivot.values[~np.isnan(pivot.values)] if pivot.size else np.array([0])
            vmax_raw     = float(np.nanmax(raw_vals)) if raw_vals.size else 1.0
            tick_targets = [0.0, vmax_raw * 0.01, vmax_raw * 0.1, vmax_raw * 0.5, vmax_raw]
            tick_vals    = [float(np.log1p(t)) for t in tick_targets]
            tick_text    = [f"{t:.2f}" if t < 10 else f"{t:.0f}" for t in tick_targets]

            fig_heat = px.imshow(
                pivot_log, color_continuous_scale="RdYlGn_r",
                labels={"color": "AE Error (log scale)"},
                title=f"Top-30 nodes by peak AE score — {cluster_ts} (May 23)",
                aspect="auto"
            )
            fig_heat.update_traces(
                hovertemplate="Node: %{y}<br>Hour: %{x}<br>AE Error: %{customdata:.4f}<extra></extra>",
                customdata=pivot.values
            )
            fig_heat.update_layout(
                height=550,
                coloraxis_colorbar=dict(tickvals=tick_vals, ticktext=tick_text, title="AE Error"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption(
                f"Color is on a log scale so subtle node-level anomalies stay visible "
                f"even when a single node spikes (max AE error today: **{vmax_raw:,.2f}**). "
                "Red bands mark hours where a node's reconstruction error exceeded normal behaviour."
            )
        else:
            st.info("No May 23 test scores available in the score file.")

        # Raw score table
        with st.expander("Show raw score data"):
            # Build ordered column list, deduplicating (node_col may equal 'instance')
            _raw_candidates = [node_col, "instance", "timestamp", ae_col, "ae_flag",
                               "if_score", "if_flag"]
            _seen = set()
            display_cols = []
            for _c in _raw_candidates:
                if _c and _c in score_df.columns and _c not in _seen:
                    display_cols.append(_c)
                    _seen.add(_c)
            st.dataframe(score_df[display_cols].head(500), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: LIVE ANOMALY PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("🚨 Live Anomaly Injection Playground")
    st.caption("Inject synthetic hardware faults in real-time and watch the Autoencoder respond.")
    st.info(
        "**How it works:** The trained Autoencoder learned to perfectly reconstruct *normal* "
        "hardware states. When you inject a fault below, it creates a data point the model has "
        "never seen — causing a high reconstruction error that triggers an anomaly alert."
    )

    # Check if models are available
    col_pg1, col_pg2 = st.columns([3,1])
    with col_pg1:
        cluster_pg = st.selectbox("Select Farm", clusters, index=0, key="pg_cluster")
    with col_pg2:
        mod_pg = st.selectbox("Modality", AVAIL_MODALITIES,
                               format_func=lambda x: MODALITY_LABELS.get(x, x),
                               key="pg_modality") if IS_V6 else "memory_slurm"
    prefix_pg = {"memory_slurm": "", "cpu": "cpu_", "disk": "disk_"}.get(mod_pg, "")
    ae_path  = MODELS_DIR / f"{prefix_pg}autoencoder_{cluster_pg}.joblib"
    sc_path  = MODELS_DIR / f"{prefix_pg}scaler_{cluster_pg}.joblib"
    iso_path = MODELS_DIR / f"{prefix_pg}isoforest_{cluster_pg}.joblib"

    if not (ae_path.exists() and sc_path.exists() and iso_path.exists()):
        st.error(
            f"Model files not found in `{MODELS_DIR}`. "
            "Make sure the `.joblib` files from training are present."
        )
        st.stop()

    # Real threshold from JSON (if available)
    real_threshold = thresholds_data.get(cluster_pg, {}).get("threshold", None)
    train_mean_err = thresholds_data.get(cluster_pg, {}).get("train_mean_error", None)
    threshold_pct  = thresholds_data.get(cluster_pg, {}).get("percentile", 99)

    # Warn if threshold percentile is unusually low (farm tuned for high recall, low precision)
    if real_threshold and train_mean_err:
        headroom_pct = (real_threshold - train_mean_err) / real_threshold * 100
        if threshold_pct < 99:
            st.warning(
                f"⚠️ **{cluster_pg}** uses an aggressive **{threshold_pct}th-percentile** threshold "
                f"(headroom: {headroom_pct:.1f}% above training mean). "
                "This farm is tuned for high recall — minor faults will always be detected. "
                "Switch to another farm for a more gradual playground experience."
            )
        else:
            st.success(
                f"Using **real trained threshold** for {cluster_pg}: "
                f"{real_threshold:.6f} ({threshold_pct}th percentile of training error)"
            )
    else:
        st.warning(
            "Real threshold not available — using 5× severity heuristic. "
            "Run `save_thresholds.py` to fix this."
        )

    st.markdown("---")
    col_ctrl, col_result = st.columns([1, 1])

    with col_ctrl:
        st.subheader("⚙️ Control Panel")
        st.markdown("### Inject Hardware Faults")
        st.markdown("*Drag sliders to simulate failures. 0 = perfectly normal.*")

        cpu_spike      = st.slider("🔥 CPU Spike (×σ above mean)",      0.0, 15.0, 0.0, 0.5,
                                   help="Simulates CPU overload — all nodes pegged at 100%")
        mem_leak       = st.slider("💧 Memory Leak (×σ above mean)",     0.0, 15.0, 0.0, 0.5,
                                   help="Simulates a runaway job eating RAM")
        io_thrash      = st.slider("💾 I/O Thrashing (×σ above mean)",   0.0, 15.0, 0.0, 0.5,
                                   help="Disk I/O saturation, swap storms")
        job_flood      = st.slider("🌊 Job Flood (×σ above mean)",        0.0, 15.0, 0.0, 0.5,
                                   help="Too many SLURM jobs submitted at once")
        gaussian_noise = st.slider("📡 Sensor Noise (std)",               0.0, 0.5,  0.0, 0.01,
                                   help="Background measurement jitter")
        run_btn = st.button("▶ Run Inference", type="primary", use_container_width=True)

    with col_result:
        st.subheader("📈 Live Model Output")

        if run_btn:
            with st.spinner("Loading model and running inference..."):
                ae, sc, iso = load_models(cluster_pg, mod_pg)
                n_features  = sc.n_features_in_

                # ── Build a true "normal" baseline ────────────────────────────
                # For ReconAE (CPU/Disk): use the AE's own internal scaler mean
                # so the baseline reconstructs cleanly (not the pipeline scaler mean).
                # For legacy sklearn AE (Memory+SLURM): use inverse_transform of zeros.
                if hasattr(ae, "sc"):  # ReconAE with internal StandardScaler
                    normal_raw = ae.sc.inverse_transform(
                        np.zeros((1, ae.sc.n_features_in_))
                    )
                else:
                    normal_raw = sc.inverse_transform(np.zeros((1, n_features)))

                # ── Inject faults into core feature signatures ────────────────
                # Each slider targets only the PRIMARY features of that fault type.
                # For memory_slurm: inject in SCALED space (+N sigma) so the
                # magnitude is always comparable regardless of raw feature scale.
                # For CPU/Disk (ReconAE): shift in AE's own sigma space.
                INJECT_SCALE = 1.5

                fault_row = normal_raw.copy()

                if mod_pg == "memory_slurm" and hasattr(sc, "feature_names_in_"):
                    fnames_pg  = list(sc.feature_names_in_)

                    mem_core_pg  = np.array([i for i, n in enumerate(fnames_pg)
                        if any(k in n for k in ["MemFree", "Active_anon", "Mapped"])])
                    io_core_pg   = np.array([i for i, n in enumerate(fnames_pg)
                        if any(k in n for k in ["SwapFree", "SwapCached", "Writeback"])])
                    cpu_core_pg  = np.array([i for i, n in enumerate(fnames_pg)
                        if n.startswith("cpu_")])
                    slurm_core_pg = np.array([i for i, n in enumerate(fnames_pg)
                        if n.startswith("status_") or n == "transition_rate_15min"])

                    # Inject in SCALED space to avoid raw-unit magnitude blowup.
                    # We scale the fault_row, add N-sigma perturbation, then invert.
                    scaled_fault = sc.transform(normal_raw)
                    if mem_leak > 0 and len(mem_core_pg) > 0:
                        scaled_fault[0, mem_core_pg]   += mem_leak  * INJECT_SCALE
                    if io_thrash > 0 and len(io_core_pg) > 0:
                        scaled_fault[0, io_core_pg]    += io_thrash * INJECT_SCALE
                    if cpu_spike > 0 and len(cpu_core_pg) > 0:
                        scaled_fault[0, cpu_core_pg]   += cpu_spike * INJECT_SCALE
                    if job_flood > 0 and len(slurm_core_pg) > 0:
                        scaled_fault[0, slurm_core_pg] += job_flood * INJECT_SCALE
                    # Convert back to raw space for the rest of the pipeline
                    fault_row = sc.inverse_transform(scaled_fault)
                else:
                    # CPU / Disk (ReconAE): shift in AE's own sigma space
                    combined = (cpu_spike + mem_leak + io_thrash + job_flood) / 4.0
                    if combined > 0 and hasattr(ae, "sc"):
                        fault_row += ae.sc.scale_ * combined * INJECT_SCALE

                if gaussian_noise > 0:
                    fault_row += np.random.normal(0, gaussian_noise, fault_row.shape) * fault_row




                scaled_normal = sc.transform(normal_raw)
                scaled_fault  = sc.transform(fault_row)

                # ── Reconstruction error ──────────────────────────────────────
                # ReconAE (cpu/disk) is uniquely identified by its internal 'sc' scaler.
                # Its score(X_raw) takes unscaled input and returns (errors, flags).
                # sklearn MLPRegressor (memory_slurm) also has score() but requires
                # (X, y) — use predict() on pipeline-scaled input instead.
                if hasattr(ae, "sc"):   # ReconAE — cpu / disk
                    err_normal_arr, _ = ae.score(normal_raw)
                    err_fault_arr,  _ = ae.score(fault_row)
                    err_normal = float(np.mean(err_normal_arr))
                    err_fault  = float(np.mean(err_fault_arr))
                else:                   # sklearn MLPRegressor — memory_slurm
                    pred_normal = ae.predict(scaled_normal)
                    pred_fault  = ae.predict(scaled_fault)
                    err_normal = float(np.mean(np.square(scaled_normal - pred_normal)))
                    err_fault  = float(np.mean(np.square(scaled_fault  - pred_fault)))

                # Use real trained threshold when available, else fall back to 5× heuristic.
                # Special case: if the normal baseline itself exceeds the threshold (farm19
                # with 85th-pct threshold), the threshold is too tight for playground use —
                # fall back to heuristic so baseline is always shown as NORMAL.
                _use_real_thr = (
                    real_threshold is not None and
                    err_normal < real_threshold  # baseline must be below threshold
                )
                if _use_real_thr:
                    is_anomaly = err_fault > real_threshold
                    threshold_label = f"Trained {threshold_pct}th-pct threshold ({real_threshold:.5f})"
                else:
                    severity   = err_fault / max(err_normal, 1e-9)
                    is_anomaly = severity > 5.0
                    threshold_label = "5× severity heuristic (approximate)"
                    if real_threshold is not None:
                        # Explain why we fell back
                        st.info(
                            f"ℹ️ **{cluster_pg}** threshold ({real_threshold:.5f}, {threshold_pct}th pct) "
                            f"is tighter than the computed baseline error ({err_normal:.5f}). "
                            "Using 5× heuristic for a meaningful playground display."
                        )

                # IsolationForest for CPU/Disk was trained on PCA-reduced features.
                # Load PCA if available and apply it before predict().
                _pca_path = MODELS_DIR / f"{prefix_pg}pca_{cluster_pg}.joblib"
                if _pca_path.exists():
                    import joblib as _jl
                    _pca = _jl.load(_pca_path)
                    _iso_input = _pca.transform(scaled_fault)
                else:
                    _iso_input = scaled_fault
                iso_result = iso.predict(_iso_input)[0]
                iso_label  = "🚨 ANOMALY" if iso_result == -1 else "✅ NORMAL"

            # ── Display Results ────────────────────────────────────────────────
            st.markdown(f"#### Reconstruction Error  \n`Threshold method: {threshold_label}`")
            err_col1, err_col2, err_col3 = st.columns(3)
            err_col1.metric("Normal Baseline Error", f"{err_normal:.5f}")
            err_col2.metric("Injected Fault Error",  f"{err_fault:.5f}",
                            delta=f"+{err_fault-err_normal:.5f}", delta_color="inverse")
            if real_threshold is not None:
                err_col3.metric("Anomaly Threshold", f"{real_threshold:.5f}")
            else:
                err_col3.metric("Error Ratio", f"{err_fault/max(err_normal,1e-9):.1f}×")

            if real_threshold is not None:
                # Progress bar: error relative to threshold
                ratio = err_fault / max(real_threshold, 1e-9)
                st.progress(min(ratio, 1.0))
                st.caption(f"Error is {ratio:.1f}× the anomaly threshold")
            else:
                severity = err_fault / max(err_normal, 1e-9)
                if severity > 1:
                    st.progress(min(severity / 20.0, 1.0))

            st.markdown("---")
            st.markdown("#### 🤖 Autoencoder Verdict")
            if is_anomaly:
                if real_threshold is not None:
                    st.error(
                        f"## 🚨 ANOMALY DETECTED\n"
                        f"Reconstruction error **{err_fault:.5f}** exceeds trained threshold "
                        f"**{real_threshold:.5f}** — this node would be flagged.",
                        icon="🚨"
                    )
                else:
                    st.error(
                        f"## 🚨 ANOMALY DETECTED\n"
                        f"Reconstruction error is approximately **{err_fault/max(err_normal,1e-9):.1f}×** "
                        f"higher than the normal baseline.",
                        icon="🚨"
                    )
            else:
                st.success(
                    "## ✅ NORMAL\nReconstruction error is within the expected range "
                    "for healthy hardware.", icon="✅"
                )

            st.markdown("#### 🌲 Isolation Forest Verdict")
            if iso_result == -1:
                st.error(f"**{iso_label}** — Isolation Forest independently confirms anomaly.", icon="🚨")
            else:
                st.success(f"**{iso_label}** — Isolation Forest sees no anomaly.", icon="✅")

            st.markdown("---")
            st.markdown("#### 📊 Error Profile vs Threshold")
            chart_vals = {"Normal Baseline": err_normal, "Injected Fault": err_fault}
            if real_threshold is not None:
                chart_vals["Anomaly Threshold"] = real_threshold
            if train_mean_err is not None:
                chart_vals["Train Mean Error"] = train_mean_err

            chart_df = pd.DataFrame.from_dict(
                {"Reconstruction Error": chart_vals}
            )
            fig_bar = px.bar(
                chart_df.reset_index().rename(columns={"index": "State"}),
                x="State", y="Reconstruction Error",
                color="State",
                color_discrete_map={
                    "Normal Baseline": "green",
                    "Injected Fault": "red",
                    "Anomaly Threshold": "orange",
                    "Train Mean Error": "blue"
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown("""
👈 **Use the control panel on the left to begin.**

Try these scenarios:

| Scenario | Setting |
|---|---|
| Mild CPU pressure | CPU Spike = 3 |
| Memory leak crash | Memory Leak = 10 |
| Total node failure | All sliders maxed |
| Sensor jitter only | Noise = 0.05 |

Click **▶ Run Inference** after adjusting sliders to see the model respond instantly.
            """)
