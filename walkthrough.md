# Walkthrough: Anomaly-Detection-Aware Split Rework (April 10, 2026)

## Context

This project builds an **unsupervised anomaly detection pipeline** for the JLab compute cluster digital twin.
The dataset (May 19–23, 2023) contains a **real anomalous event on May 23** that required JLab IT intervention.
The data is **unlabelled** — the exact affected nodes and anomaly start-time are unknown.

---

## Critical Bug Fixed: Wrong Split Design

The previous 70/15/15 chronological split was **fundamentally incorrect** for this use case:

| Problem | Impact |
|---------|--------|
| Val/test sets included May 23 anomalous data | Model cannot be tuned without contaminating the anomaly signal |
| Train set included some May 23 data | Normal-behaviour baseline was polluted with anomalous data |
| Uniform 5min resample for all modalities | `cpu_data` collapsed to only 12 training points — far too sparse |
| No awareness of unsupervised context | Validation split is meaningless with no labels |

---

## 1. New Split Pipeline ([make_splits.py](file:///d:/SamsungPrismCrosslayerRAN/eda/make_splits.py))

Redesigned around the **anomaly detection constraint**: train must contain ONLY normal data.

```
TRAIN = May 19 00:00 → May 22 23:59  (normal cluster behaviour)
TEST  = May 23 00:00 → May 23 23:59  (real anomalous event)
VAL   = NONE  (unsupervised — no labels exist)
```

### Per-Modality Resampling

| Modality | Resample | Rationale |
|----------|----------|-----------|
| `cpu_data` | 5T (5-min) | ~15s scrape → collapse to 5-min windows |
| `disk_data` | 5T (5-min) | I/O counters need time-window aggregation |
| `memory_data` | 1T (1-min) | Higher-frequency signals preserved |
| `slurm_data` | None | Event-driven scheduler — do not resample |

Gap fill: forward-fill up to **3 consecutive NaN steps**, then drop remaining NaN rows.

### Hardware Config Column

A new `hw_config` column is inferred from node/instance label patterns (5 JLab hardware configs).
Currently all rows are tagged `generic` (detailed node naming not present in this dataset subset).

### Outputs

```
artifacts/splits/
  cpu_data_train.parquet       (5,976 rows)
  cpu_data_test.parquet        (0 rows — collection gap May 23)
  disk_data_train.parquet      (190,643 rows)
  disk_data_test.parquet       (0 rows — collection gap May 23)
  memory_data_train.parquet    (1,827,271 rows, 82.9%)
  memory_data_test.parquet     (377,660 rows, 17.1%)
  slurm_data_train.parquet     (3,709,122 rows, 83.1%)
  slurm_data_test.parquet      (754,232 rows,  16.9%)
  split_manifest.json          (full provenance record)
```

### Sanity Checks Passed

- `memory_data`: zero temporal overlap ✓ (`max_train=2023-05-22 23:59:59` < `min_test=2023-05-23 00:00:00`)
- `slurm_data`:  zero temporal overlap ✓ (`max_train=2023-05-22 23:59:47` < `min_test=2023-05-23 00:00:47`)

---

## 2. Known Pre-existing Data Issue

`cpu_data` and `disk_data` have **0 test rows** — their Prometheus scraper stopped collecting
before May 23, creating an 84% temporal gap. This is a dataset collection limitation.

**Recommendation**: Focus anomaly detection on `memory_data` and `slurm_data` — the only
modalities with complete coverage of the anomalous May 23 event.

---

## 3. Updated Train Stats ([compute_train_stats.py](file:///d:/SamsungPrismCrosslayerRAN/eda/compute_train_stats.py))

Fixed the train boundary to `2023-05-22 23:59:59` (was previously a fractional 70% split).
Stats are now computed strictly on **normal-behaviour data only**:

| Modality | File | Train Rows | Mean | Std |
|----------|------|----------|------|-----|
| cpu_data | `node_cpu_seconds_total_0` | 15,966,720 | 26,286 | 48,699 |
| disk_data | `node_disk_io_now_0` | 1,503,057 | 1.77 | 15.02 |
| memory_data | `node_memory_Active_anon_bytes` | 1,826,802 | 25.5 GB | 31.3 GB |
| slurm_data | `slurm_node_cpu_alloc` | 1,854,561 | 59.58 | 55.73 |

---

## 4. Updated EDA HTML Report ([build_eda.py](file:///d:/SamsungPrismCrosslayerRAN/eda/build_eda.py))

Completely rewritten with a **dark-themed sidebar-navigation interface** including:
- **Split Design section**: visual train/test bar + boundary table with anomaly context
- **Resample Policy table**: per-modality rules and rationale
- **Pipeline Dataflow**: end-to-end `raw → clean → split → stats → train` diagram
- **Data Quality table**: missing rate badges (green/yellow/red) per modality

---

## 5. How to Run the Full Pipeline

```powershell
# Activate environment
.\.venv\Scripts\activate

# Step 1: Structural cleaning (if not already done)
python -m eda.clean --all-folders --dataset-dir dataset --out-dir structural_clean

# Step 2: NEW - Correct splits (binary train/test, per-modality resample)
python -m eda.make_splits --clean-dir structural_clean --out-dir artifacts/splits

# Step 3: Train statistics (normal data only, no leakage)
python -m eda.compute_train_stats --clean-dir structural_clean --out artifacts/eda/train_stats.json

# Step 4: Enrich split parquets with is_outlier + value_norm
python -m eda.apply_stats --splits-dir artifacts/splits --stats artifacts/eda/train_stats.json

# Step 5: Rebuild EDA HTML report
python -m eda.build_eda --dataset-dir dataset

# Step 6: TS analysis (ADF + ACF/PACF on train split)
python -m eda.ts_analysis --clean-dir structural_clean
```

---

## 6. Next Steps

1. **Anomaly model training**: Use `memory_data_train.parquet` + `slurm_data_train.parquet`
   as the normal-behaviour training corpus.
2. **Model selection**: Consider Isolation Forest, Autoencoder, or LSTM-AE for multivariate anomaly detection.
3. **Multimodal join**: Join memory + slurm splits on `(node, timestamp)` with 1-min resolution
   (round slurm timestamps to nearest minute before joining).
4. **Evaluation**: Test model predictions against `*_test.parquet` — anomalous behaviour expected
   somewhere in the May 23 window.
