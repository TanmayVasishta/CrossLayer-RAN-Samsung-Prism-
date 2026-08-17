"""Project-wide audit: data integrity, JSON validity, model loadability, schema checks."""
import pandas as pd
import numpy as np
import json
import joblib
from models.autoencoder import ReconAE  # noqa: F401 — required for CPU autoencoder unpickling
from eda.cpu_pipeline_v3 import ReconAE  # noqa: F811 — fallback for old pickles
import time
from pathlib import Path

BASE = Path(".")
errors = []
warnings = []

def banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ─── 1. Parquet Data Integrity ────────────────────────────────────────────────
banner("1. PARQUET DATA INTEGRITY")

parquet_checks = [
    ("artifacts/splits/cpu_data_train.parquet",
     ["timestamp","instance","hw_config","cpu_idle_rate","cpu_user_rate"], 1_000_000),
    ("artifacts/splits/cpu_data_test.parquet",
     ["timestamp","instance","hw_config","cpu_idle_rate","cpu_user_rate"], 400_000),
    ("artifacts/splits/memory_data_train.parquet",
     ["timestamp","instance","hw_config"], 100_000),     # memory uses 'instance' not 'node'
    ("artifacts/splits/memory_data_test.parquet",
     ["timestamp","instance","hw_config"], 10_000),
    ("artifacts/splits/slurm_data_train.parquet",
     ["timestamp","node","hw_config"], 1_000),
    ("artifacts/splits/slurm_data_test.parquet",
     ["timestamp","node","hw_config"], 100),
    ("artifacts/features/cpu_data_train_features.parquet",
     ["timestamp","instance","hw_config"], 1_000_000),
    ("artifacts/features/cpu_data_test_features.parquet",
     ["timestamp","instance","hw_config"], 400_000),
    ("artifacts/features/memory_data_train_features.parquet",
     ["timestamp","instance","hw_config"], 100_000),     # memory uses 'instance'
    ("artifacts/features/slurm_data_test_features.parquet",
     ["timestamp","hw_config","is_distress_status"], 100),
    ("artifacts/models/cpu_anomaly_scores.parquet",
     ["instance","timestamp","hw_config","if_score","ae_score","split"], 400_000),
]

for path_str, required_cols, min_rows in parquet_checks:
    p = BASE / path_str
    name = p.name
    try:
        df = pd.read_parquet(p)
        missing_cols = [c for c in required_cols if c not in df.columns]
        num_cols = df.select_dtypes("number")
        nan_pct = df.isna().mean().mean() * 100
        has_inf = np.isinf(num_cols.values).any() if not num_cols.empty else False
        n = len(df)

        issues = []
        if missing_cols: issues.append(f"MISSING COLS: {missing_cols}")
        if has_inf:       issues.append("HAS INF VALUES")
        if n == 0:        issues.append("EMPTY FILE")
        if n < min_rows:  issues.append(f"ROW COUNT LOW: {n:,} < {min_rows:,}")

        if issues:
            status = "WARN"
            for iss in issues:
                warnings.append(f"{name}: {iss}")
        else:
            status = "OK"
        detail = f"{n:,} rows | {len(df.columns)} cols | NaN={nan_pct:.1f}%"
        if issues:
            detail += " | " + " | ".join(issues)
        print(f"  {status:<5} {name:<50} {detail}")
    except Exception as e:
        print(f"  ERROR {name}: {e}")
        errors.append(f"{name}: {e}")

# ─── 2. JSON Validity ─────────────────────────────────────────────────────────
banner("2. JSON FILE VALIDITY")

jsons = [
    "artifacts/models/cpu_training_summary.json",
    "artifacts/models/cpu_evaluation_report.json",
    "artifacts/models/evaluation_report_metrics_v4.json",
    "artifacts/models/evaluation_report_metrics_v5.json",
    "artifacts/models/multimodel_results_v4.json",
    "artifacts/models/multimodel_results_v5.json",
    "artifacts/models/thresholds.json",
    "artifacts/splits/split_manifest.json",
    "artifacts/features/feature_manifest.json",
]

for path_str in jsons:
    p = BASE / path_str
    try:
        with open(p) as f:
            d = json.load(f)
        n_keys = len(d) if isinstance(d, dict) else len(d)
        print(f"  OK    {p.name:<50} {n_keys} top-level keys")
    except FileNotFoundError:
        print(f"  MISS  {p.name}")
        warnings.append(f"JSON missing: {p.name}")
    except Exception as e:
        print(f"  ERROR {p.name}: {e}")
        errors.append(f"JSON {p.name}: {e}")

# ─── 3. Model Loadability ─────────────────────────────────────────────────────
banner("3. MODEL FILES LOADABILITY")

farms = ["farm14", "farm16", "farm18", "farm19", "farm23"]
model_prefixes = {
    "Memory/SLURM": ["isoforest", "autoencoder", "pca", "scaler"],
    "CPU":          ["cpu_isoforest", "cpu_autoencoder", "cpu_pca", "cpu_scaler"],
}

for modality, prefixes in model_prefixes.items():
    print(f"\n  [{modality}]")
    for farm in farms:
        row = []
        for prefix in prefixes:
            path = BASE / f"artifacts/models/{prefix}_{farm}.joblib"
            try:
                t0 = time.perf_counter()
                obj = joblib.load(path)
                t1 = time.perf_counter()
                size_kb = round(path.stat().st_size / 1024, 0)
                row.append(f"{prefix.split('_')[-1]}({size_kb:.0f}KB)")
            except FileNotFoundError:
                row.append(f"{prefix}=MISSING")
                errors.append(f"Model missing: {prefix}_{farm}.joblib")
            except Exception as e:
                row.append(f"{prefix}=ERROR:{e}")
                errors.append(f"Model load error {prefix}_{farm}: {e}")
        print(f"    {farm}: " + "  ".join(row))

# ─── 4. Evaluation Metrics Completeness ───────────────────────────────────────
banner("4. EVALUATION METRICS COMPLETENESS")

required_metrics = ["precision", "recall", "f1_score", "robust_recall", "latency_ms_per_row"]

for json_path, label in [
    ("artifacts/models/evaluation_report_metrics_v4.json", "Memory+SLURM (v4)"),
    ("artifacts/models/evaluation_report_metrics_v5.json", "All modalities (v5)"),
]:
    p = BASE / json_path
    try:
        with open(p) as f:
            d = json.load(f)

        # v5 has nested structure
        if "by_modality" in d:
            farm_dict = d["by_modality"].get("memory_slurm", {})
            cpu_dict  = d["by_modality"].get("cpu", {})
        else:
            farm_dict = d
            cpu_dict  = {}

        print(f"\n  {label}")
        for modality, mdata in [("memory_slurm", farm_dict), ("cpu", cpu_dict)]:
            if not mdata:
                continue
            for farm, models in mdata.items():
                for model_name, metrics in models.items():
                    missing_m = [m for m in required_metrics if m not in metrics]
                    if missing_m:
                        msg = f"MISSING metrics in {label}/{modality}/{farm}/{model_name}: {missing_m}"
                        warnings.append(msg)
                        print(f"    WARN  {farm}/{model_name}: missing {missing_m}")
                    else:
                        print(f"    OK    {farm}/{model_name} [{modality}]: all {len(required_metrics)} metrics present")
    except Exception as e:
        print(f"  ERROR reading {json_path}: {e}")
        errors.append(str(e))

# ─── 5. Date Range Validation ────────────────────────────────────────────────
banner("5. DATE RANGE VALIDATION (train=May19-22, test=May23)")

date_checks = [
    ("artifacts/splits/cpu_data_train.parquet", "timestamp", "2023-05-19", "2023-05-22", "train"),
    ("artifacts/splits/cpu_data_test.parquet",  "timestamp", "2023-05-23", "2023-05-23", "test"),
]

for path_str, ts_col, exp_start_date, exp_end_date, split in date_checks:
    p = BASE / path_str
    try:
        df = pd.read_parquet(p, columns=[ts_col])
        df[ts_col] = pd.to_datetime(df[ts_col])
        actual_start = df[ts_col].min().date().isoformat()
        actual_end   = df[ts_col].max().date().isoformat()
        ok = actual_start >= exp_start_date and actual_end <= exp_end_date
        status = "OK" if ok else "WARN"
        print(f"  {status}    {p.name} ({split}): {actual_start} → {actual_end}")
        if not ok:
            warnings.append(f"Date range mismatch in {p.name}: {actual_start}→{actual_end}")
    except Exception as e:
        print(f"  ERROR {p.name}: {e}")
        errors.append(str(e))

# ─── Summary ──────────────────────────────────────────────────────────────────
banner("AUDIT SUMMARY")
print(f"  Errors   : {len(errors)}")
for e in errors:
    print(f"    [ERR]  {e}")
print(f"  Warnings : {len(warnings)}")
for w in warnings:
    print(f"    [WARN] {w}")
if not errors and not warnings:
    print("  All checks passed! Project is healthy.")
