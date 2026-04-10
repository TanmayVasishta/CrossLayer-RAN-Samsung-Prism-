## CrossLayerAI-RAN (local EDA + splits)

You have a large Prometheus-style metrics dataset under `dataset/`:

- `dataset/cpu_data/*.csv.gz`
- `dataset/disk_data/*.csv.gz`
- `dataset/memory_data/*.csv.gz`
- `dataset/slurm_data/*.csv.gz`

This repo adds an EDA pipeline that:

- Scans large `.csv.gz` files **in chunks** (no full-load required).
- Builds a **local HTML report** with key plots and summary tables.
- Runs a **local-host dashboard** for interactive exploration.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Build EDA artifacts + HTML report

This scans the dataset and writes:

- `artifacts/eda/summary.json`
- `reports/eda_report.html`

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

## Step 3: Data preparation + time-based splitting (next)

Once you’re happy with EDA, we’ll generate:

- a normalized, aligned time series table per node (common granularity)
- train/val/test splits with **time leakage protection**

