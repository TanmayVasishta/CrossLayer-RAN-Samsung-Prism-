# 📡 CrossLayerAI-RAN

**SRIB-PRISM Program · Worklet 26NCOAM02BMS**

Cross-layer, infrastructure-aware AI for proactive RAN / HPC anomaly
detection. An unsupervised multi-model pipeline trained on Jefferson Lab HPC
telemetry (CPU, Memory, Disk, SLURM) to detect hardware distress events
without ground-truth labels.

> 🚀 **Live interactive demo:** _deploy your own from this repo — see
> [Deploying the demo](#deploying-the-demo) below_
>
> 📊 **Instant static preview:** [`demo/`](demo/), served via GitHub Pages —
> no cold start, always available
>
> 📋 **[Full Project Status & Results](PROJECT_STATUS.md)** — detailed
> per-farm metrics, known limitations, and a prioritized improvement list
>
> 🏗️ See [`project_overview.html`](project_overview.html) for a full project
> tour aimed at teammates (problem, dataset, pipeline, models, results)

---

## Results at a glance (v7)

| Farm | Best Result | Modality |
|------|-------------|----------|
| farm18 | **100.0%** recall (Autoencoder) | Memory+SLURM |
| farm16 | 98.0% recall (LOF) | Memory+SLURM |
| farm19 | 96.2% recall (Ensemble) | CPU |
| farm14 | 78.2% recall (Autoencoder) | Memory+SLURM |
| farm23 | No distress events in test window | — |

Precision is low across the board (single-digit percent in most cases) — this
is an open problem for an unsupervised detector evaluated against one
24-hour labeled event, not a bug. Full breakdown, per-model tables, and the
reasoning behind it: [PROJECT_STATUS.md](PROJECT_STATUS.md#4-model-results-v7--pulled-directly-from-artifactsmodelsevaluation_report_metrics_v7json).

## Pipeline Overview

| Phase | Module | Description |
|-------|--------|--------------|
| 1 | `eda/clean.py` | Chunk-safe structural cleaning (drop junk cols, parse timestamps, dedup) |
| 2 | `eda/build_eda.py` | HTML EDA report with data quality analysis |
| 3 | `eda/make_splits.py` | Chronological split: Train=May 19-22, Test=May 23 |
| 4 | `eda/compute_train_stats.py` | Train-only IQR + normalization stats |
| 5 | `eda/apply_stats.py` | Enrich data with `is_outlier` + `value_norm` |
| 6 | `eda/ts_analysis.py` | ADF stationarity + ACF/PACF analysis |
| 7 | `eda/feature_eng.py` | Rolling windows + lag diffs + status encoding |
| 8 | `eda/multimodal_join.py` | Memory + SLURM multimodal join |
| 9 | `eda/multimodel_anomaly.py` | IF + LOF + AE + Ensemble per farm |
| — | `eda/cpu_pipeline_v3.py`, `eda/disk_pipeline.py` | Standalone CPU/Disk modality pipelines |

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

The 180GB+ raw dataset (from [Zenodo](https://zenodo.org/records/10058230))
is not checked into this repo. Everything from `demo_app.py` down to the
`eda/build_eda.py` report can run directly off the `artifacts/` already
checked in — you only need the raw `dataset/` folder if you're re-running
the pipeline from scratch.

## Step 1: Build EDA artifacts + HTML report

```bash
python -m eda.build_eda --dataset-dir dataset
```

Optional knobs (useful because the dataset is huge):

```bash
python -m eda.build_eda --dataset-dir dataset --max-files-per-folder 10 --sample-rows 200000
```

## Step 2: Run the local EDA website

```bash
streamlit run eda/eda_app.py
```

## Step 3: Data preparation + chronological splitting

```bash
python -m eda.clean --all-folders --dataset-dir dataset --out-dir structural_clean
python -m eda.make_splits --clean-dir structural_clean --out-dir artifacts/splits
python -m eda.compute_train_stats --clean-dir structural_clean --out artifacts/eda/train_stats.json
python -m eda.apply_stats --splits-dir artifacts/splits --stats artifacts/eda/train_stats.json
```

## Step 4: Feature engineering + multimodel anomaly detection

```bash
python -m eda.feature_eng --splits-dir artifacts/splits --out-dir artifacts/features
python -m eda.multimodal_join --features-dir artifacts/features
python -m eda.multimodel_anomaly --features-dir artifacts/features --out-dir artifacts/models
```

## Step 5: Demo

3-tab Streamlit demo (Results Dashboard / Time-Series Scores / Live Anomaly
Playground):

```bash
streamlit run demo_app.py
```

## Deploying the demo

To get your own always-updatable live link (rather than depending on
someone else's deployment):

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. **New app** → pick this repo → branch `main` → main file `demo_app.py`.
3. Deploy. First load will be slow (installing pinned dependencies + loading
   ~160MB of model artifacts); subsequent loads are fast until the app sleeps
   from inactivity (Streamlit Community Cloud free-tier behavior — upgrading
   to a paid tier or pinging it periodically keeps it warm).
4. Update the link at the top of this README once it's live.

## Testing

```bash
pip install pytest
pytest
```

## Project structure, full results tables, known limitations, and a
prioritized improvement list live in [PROJECT_STATUS.md](PROJECT_STATUS.md).
