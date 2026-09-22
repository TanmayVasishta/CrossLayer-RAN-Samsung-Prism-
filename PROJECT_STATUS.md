# 📡 CrossLayerAI-RAN — Project Status Report

**Project:** 26NCOAM02BMS — Cross-Layer AI-RAN Infrastructure-Aware AI for Proactive RAN
**Program:** SRIB-PRISM (Samsung Research Institute Bangalore)
**Institution:** B.M.S. College of Engineering
**Last Updated:** September 23, 2026
**Live Demo:** [samsungcrosslayer.streamlit.app](https://samsungcrosslayer.streamlit.app/) · instant static preview at [`demo/`](demo/), deployed via GitHub Pages
**Formal Report:** [project_report.html](project_report.html) / [project_report.docx](project_report.docx)

**Team:**

| Name | USN |
|------|-----|
| Tanmay Vasishta | 1WA23CS012 |
| Hitha Harish | 1BM23CS115 |
| Samriddhi Singh | 1BM23CS295 |
| Sinchana Hemanth | 1BM23CS330 |

---

## 1. Project Overview

Cross-layer, infrastructure-aware AI for proactive RAN/HPC anomaly detection. An **unsupervised multi-model pipeline** trained on Jefferson Lab (JLab) HPC telemetry data (CPU, Memory, Disk, SLURM) to detect hardware distress events **without ground-truth labels**.

### Key Facts

| Item | Detail |
|------|--------|
| **Dataset** | 180 GB+ JLab Prometheus-style HPC metrics (May 19–23, 2023) |
| **Modalities** | CPU, Memory, Disk, SLURM (4 telemetry sources) |
| **Hardware Clusters** | 5 farms: farm14, farm16, farm18, farm19, farm23 |
| **Real Anomaly Event** | May 23, 2023 — JLab IT-confirmed hardware distress |
| **Approach** | Fully unsupervised (no labels) |
| **Models** | Isolation Forest, LOF, Autoencoder (MLP), Ensemble (majority vote) |
| **Best Result** | 100% recall on farm18 (Autoencoder, Memory+SLURM), 96.2% on farm19 (Ensemble, CPU) |
| **Results version** | v7 (`AE_Adaptive + Ens_Adaptive + LOF for CPU and Disk`), generated 2026-07-30 |

---

## 2. Pipeline Architecture

```
Raw .csv.gz  →  Structural Clean  →  Chronological Split  →  Train Stats  →  Apply Stats
    ↓                                      ↓                      ↓
 EDA Report                         Feature Engineering     Stationarity
                                           ↓               (ADF + ACF/PACF)
                                    Multimodal Join
                                           ↓
                                   Multimodel Training
                                  (IF + LOF + AE + Ensemble)
                                           ↓
                                    Evaluation & Demo
```

### Pipeline Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `eda/clean.py` | Chunk-safe structural cleaning (drop junk cols, parse timestamps, dedup) |
| 2 | `eda/build_eda.py` | Dark-themed HTML report with summary tables, distributions, outlier rates |
| 3 | `eda/make_splits.py` | Binary chronological split: Train=May 19-22 (normal), Test=May 23 (anomalous) |
| 4 | `eda/compute_train_stats.py` | IQR bounds + mean/std computed on train-only data (no leakage) |
| 5 | `eda/apply_stats.py` | Enrich data with `is_outlier` flag + `value_norm` using train-derived stats |
| 6 | `eda/ts_analysis.py` | ADF stationarity test + ACF/PACF autocorrelation analysis |
| 7 | `eda/feature_eng.py` | Rolling window features (5/15/60min) + lag diffs (memory), ratio + status encoding (SLURM) |
| 8 | `eda/multimodal_join.py` | Join memory + SLURM features on (node, timestamp) |
| 9 | `eda/multimodel_anomaly.py` | Train IF + LOF + AE + Ensemble per hw_config cluster |
| 10 | `eda/cpu_pipeline_v3.py` / `eda/cpu_evaluate.py` | CPU-modality pipeline + evaluation (Polars-accelerated) |
| 11 | `eda/disk_pipeline.py` / `eda/disk_evaluate.py` | Disk-modality pipeline + evaluation |
| 12 | `models/train_baseline.py` | Alternative baseline training (IF + AE per modality) |
| 13 | `demo_app.py` | 3-tab Streamlit demo (Dashboard, Time-Series Scores, Live Playground) |
| 14 | `demo/` | Static HTML dashboard, deployed to GitHub Pages |
| 15 | `tests/` | Unit tests — data cleaning + demo data-loading helpers |

`eda/cpu_pipeline.py` and `eda/cpu_pipeline_v2.py` are earlier iterations, kept for history under `eda/archive/` — `cpu_pipeline_v3.py` is the version everything actually uses.

---

## 3. Data Split Design

> **Critical design decision:** The split is purpose-built for unsupervised anomaly detection.

```
TRAIN : May 19 00:00 → May 22 23:59  (normal cluster behaviour ONLY)
TEST  : May 23 00:00 → May 23 23:59  (real anomalous JLab event)
VAL   : NONE — unsupervised pipeline, no labels exist
```

### Per-Modality Split Summary

| Modality | Train Rows | Test Rows | Resample | Overlap-Free |
|----------|-----------|-----------|----------|-------------|
| memory_data | 1,827,271 (82.9%) | 377,660 (17.1%) | 1min | ✅ |
| slurm_data | 3,709,122 (83.1%) | 754,232 (16.9%) | None (event-driven) | ✅ |
| cpu_data | 5,976 | 0 rows | 5min | N/A — collection gap |
| disk_data | 190,643 | 0 rows | 5min | N/A — collection gap |

### Known Data Limitation

`cpu_data` and `disk_data` have **zero May 23 coverage** due to a Prometheus scraper gap (84% temporal gap). The CPU and Disk pipelines process and train models but use **cross-referenced distress labels from SLURM** for indirect evaluation.

---

## 4. Model Results (v7 — pulled directly from `artifacts/models/evaluation_report_metrics_v7.json`)

### Memory + SLURM (Primary — Best Coverage)

| Farm | Model | Recall | Precision | F1 |
|------|-------|--------|-----------|----|
| farm14 | Autoencoder | 78.2% | 6.85% | 12.60% |
| farm16 | LOF | 98.0% | 3.05% | 5.92% |
| farm18 | Autoencoder | **100.0%** | 0.94% | 1.86% |
| farm19 | LOF | 92.1% | 0.79% | 1.56% |
| farm23 | — | 0.0% | — | — |

### CPU (Cross-Referenced with SLURM Distress) — best model per farm

| Farm | Best Model | Recall | Precision |
|------|-----------|--------|-----------|
| farm14 | LOF | 69.2% | 0.72% |
| farm16 | LOF | 89.7% | 2.28% |
| farm18 | LOF | 69.1% | 0.08% |
| farm19 | Ensemble | **96.2%** | 6.83% |
| farm23 | — | 0.0% | — |

### Disk (Cross-Referenced with SLURM Distress) — best model per farm

| Farm | Best Model | Recall | Precision |
|------|-----------|--------|-----------|
| farm14 | Isolation Forest | 39.0% | 5.67% |
| farm16 | LOF | 89.9% | 4.94% |
| farm18 | Isolation Forest | 45.2% | 0.20% |
| farm19 | LOF | 7.4% | 0.53% |
| farm23 | — | 0.0% | — |

### Key Observations

1. **No single model wins everywhere** — Autoencoder leads on farm14/farm18 (memory+SLURM), LOF leads on farm16/farm19 across modalities, Ensemble wins on farm19 CPU.
2. **Precision is universally low** (well under 10% almost everywhere) — expected for unsupervised detection against a single 24-hour labeled event with no negative-class tuning signal. This is the project's main open problem, not a bug.
3. **farm23 has zero distress events in every modality** — the test window for this farm shows no anomalous behaviour, so recall/precision are undefined (reported as 0%) rather than meaningfully bad.
4. **Latency is sub-millisecond per row** for the tree/MLP models — real-time-capable inference, verified by direct benchmark inside `eda/cpu_evaluate.py` / `eda/disk_evaluate.py`.

---

## 5. Artifacts Generated

### Reports & Documentation
- `reports/eda_report.html` — Comprehensive dark-themed EDA report
- `reports/guide_presentation.html` — Self-contained guide presentation
- `project_overview.html` — Full project tour for teammates
- `information.txt` — Running dev log of major changes/decisions across sessions

### Model Artifacts (per farm × modality)
- `artifacts/models/*.joblib` — 60 trained model files (Autoencoder, Isolation Forest, LOF, Scaler — 5 farms × 3 modalities)
- `artifacts/models/thresholds.json` — Per-cluster AE anomaly thresholds
- `artifacts/models/multimodel_results_v7.json` — Unified 3-modality results
- `artifacts/models/evaluation_report_metrics_v7.json` — Detailed evaluation metrics (source for the tables above)
- `artifacts/models/anomaly_scores_*.parquet`, `cpu_anomaly_scores.parquet`, `disk_anomaly_scores.parquet` — Per-row anomaly scores with timestamps

### Data Artifacts
- `artifacts/splits/*.parquet` — Train/test split data files
- `artifacts/splits/split_manifest.json` — Full split provenance
- `artifacts/eda/train_stats.json` — Train-only IQR + normalization stats
- `artifacts/eda/summary.json` — EDA scan summary
- `artifacts/features/feature_manifest.json` — Feature engineering manifest

---

## 6. How to Reproduce

```bash
# 1. Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Structural Cleaning
python -m eda.clean --all-folders --dataset-dir dataset --out-dir structural_clean

# 3. Chronological Splitting
python -m eda.make_splits --clean-dir structural_clean --out-dir artifacts/splits

# 4. Train-Only Statistics
python -m eda.compute_train_stats --clean-dir structural_clean --out artifacts/eda/train_stats.json

# 5. Apply Statistics (enrich with is_outlier + value_norm)
python -m eda.apply_stats --splits-dir artifacts/splits --stats artifacts/eda/train_stats.json

# 6. EDA HTML Report
python -m eda.build_eda --dataset-dir dataset

# 7. Time-Series Analysis (ADF + ACF/PACF)
python -m eda.ts_analysis --clean-dir structural_clean

# 8. Feature Engineering
python -m eda.feature_eng --splits-dir artifacts/splits --out-dir artifacts/features

# 9. Multimodal Join
python -m eda.multimodal_join --features-dir artifacts/features

# 10. Multimodel Anomaly Detection
python -m eda.multimodel_anomaly --features-dir artifacts/features --out-dir artifacts/models

# 11. Run Demo
streamlit run demo_app.py
```

The raw dataset (180GB+, from [Zenodo](https://zenodo.org/records/10058230)) is not checked into this repo — steps 2 onward assume it's present locally under `dataset/`. Everything from step 3 onward runs directly off the checked-in `artifacts/` if you just want to explore results without re-running the pipeline.

---

## 7. Potential Improvements

### 🔴 High Priority

1. **Improve Precision (reduce false positives)**
   - Current precision is 1-9% across most models — too many false alarms for production
   - Solutions: threshold tuning per farm, adaptive contamination rates, temporal smoothing (flag only if anomaly persists for N consecutive windows)

2. **Temporal Anomaly Models (LSTM-AE / Transformer)**
   - Current models treat each row independently — no temporal context
   - An LSTM Autoencoder or Transformer-based approach could capture temporal patterns
   - Expected to significantly improve precision by understanding "what's normal over time"

3. **Multivariate Anomaly Detection**
   - Currently models are per-modality; a true cross-layer model could jointly analyze CPU + Memory + Disk + SLURM
   - This is the core "cross-layer" thesis of the project

### 🟡 Medium Priority

4. **Hyperparameter Optimization**
   - IF contamination, AE architecture depth, LOF n_neighbors are currently fixed
   - A systematic sweep (even without labels — using silhouette-based or reconstruction-error-based selection) could improve results

5. **Sliding Window Anomaly Scoring**
   - Instead of scoring individual points, score sliding windows (e.g., 5-minute or 15-minute)
   - Would capture "sustained anomalies" vs. transient spikes

6. **Node-Level Aggregation**
   - Current models flag individual rows — aggregating to "anomalous nodes" would be more actionable
   - A node should be flagged if >X% of its rows in a time window are anomalous

7. **Online / Streaming Inference**
   - Package the trained models into a real-time inference API (Flask/FastAPI)
   - Could ingest live Prometheus metrics and flag anomalies in real-time

### 🟢 Nice to Have

8. **Explainability (SHAP / Feature Attribution)**
   - For each flagged anomaly, show which features contributed most
   - Critical for operational adoption — operators need to know "what failed"

9. **Drift Detection**
   - Monitor for model degradation over time (data distribution shift)
   - Alert when retraining is needed

10. **Anomaly Severity Scoring**
    - Instead of binary flag, provide a severity score (0-1)
    - Based on reconstruction error magnitude relative to threshold

11. **Evaluation on Additional Real Events**
    - Current evaluation is on a single 24-hour anomalous event (May 23)
    - Validation on additional anomaly events would strengthen confidence

12. **Docker Containerization**
    - Package the entire pipeline (data processing + model training + demo) into a Docker image
    - Ensures reproducibility across environments

---

## 8. Repository Structure

```
CrossLayer-RAN-Samsung-Prism-/
├── dataset/                     # Raw JLab HPC telemetry (180GB+, gitignored)
├── structural_clean/            # Cleaned parquets (gitignored, reproducible)
├── artifacts/
│   ├── eda/                     # EDA summaries, train stats
│   ├── features/                # Feature matrices (parquets)
│   ├── models/                  # Trained models (.joblib) + results (JSON)
│   ├── scores/                  # Anomaly score outputs
│   └── splits/                  # Train/test split parquets + manifests
├── demo/                        # Static HTML dashboard, deployed to GitHub Pages
│   ├── index.html
│   ├── app.js
│   └── style.css
├── eda/                         # Core pipeline modules
│   ├── clean.py                 # Phase 1: Structural cleaning
│   ├── build_eda.py             # Phase 2: EDA report generation
│   ├── make_splits.py           # Phase 3: Chronological splitting
│   ├── compute_train_stats.py   # Phase 4: Train-only statistics
│   ├── apply_stats.py           # Phase 5: Apply statistics globally
│   ├── ts_analysis.py           # Phase 6: Time-series + stationarity analysis
│   ├── feature_eng.py           # Phase 7: Feature engineering
│   ├── multimodal_join.py       # Phase 8: Multimodal join
│   ├── multimodel_anomaly.py    # Phase 9: Multimodel training
│   ├── cpu_pipeline_v3.py       # CPU pipeline (canonical)
│   ├── cpu_evaluate.py          # CPU evaluation
│   ├── disk_pipeline.py         # Disk pipeline
│   ├── disk_evaluate.py         # Disk evaluation
│   ├── archive/                 # Superseded pipeline versions, kept for history
│   └── lib.py                   # Shared utilities
├── models/
│   ├── autoencoder.py           # Shared ReconAE class
│   └── train_baseline.py        # IF + AE baseline training
├── tools/
│   ├── audit_project.py         # Project-wide data/model integrity audit
│   └── eval_metrics.py          # Precision/recall/F1/latency summary utility
├── tests/
│   └── test_clean_and_lib.py    # Unit tests
├── reports/
│   ├── eda_report.html          # EDA HTML report
│   └── guide_presentation.html  # Guide presentation
├── demo_app.py                  # Streamlit 3-tab demo application
├── project_overview.html        # Full project overview HTML
├── requirements.txt             # Python dependencies (pinned)
├── PROJECT_STATUS.md            # This document
└── README.md                    # Setup + usage instructions
```

---

## 9. Dependencies

Pinned in `requirements.txt` to the versions the checked-in `.joblib` models were verified against (see README for why these are pinned rather than open-ended):

```
pandas==2.3.2       numpy==2.2.6        pyarrow==21.0.0
plotly==6.8.0        streamlit==1.49.1   jinja2==3.1.6
tqdm==4.67.1          scikit-learn==1.7.1 statsmodels==0.14.6
matplotlib==3.10.5    joblib==1.5.2
```

---

*Generated: September 20, 2026*
