"""
eda/compute_train_stats.py
==========================
Computes statistical boundaries (IQR, Mean, Std) STRICTLY on the TRAIN SPLIT
of structurally cleaned data. This prevents data leakage into test.

Updated for the binary anomaly-detection split:
  - Train boundary is fixed to May 22 23:59:59 (normal data only)
  - No validation split exists
  - Reads from split_manifest.json for train time range

Output:
  artifacts/eda/train_stats.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eda.lib import ensure_dir, load_json, write_json

# Hardcoded boundary aligned with make_splits.py -- single source of truth
TRAIN_END_ISO = "2023-05-22 23:59:59.999999999"


def compute_for_file(
    parquet_path: Path,
    train_end_iso: str,
    time_col: str,
    value_col: str,
) -> Dict[str, Any]:
    """Read a structural-clean parquet, filter to train split, compute stats."""
    try:
        df = pd.read_parquet(parquet_path, columns=[time_col, value_col])
    except Exception as e:
        print(f"  [ERROR] Failed to read {parquet_path}: {e}")
        return {}

    train_end_ts = pd.to_datetime(train_end_iso)
    df_train = df[df[time_col] < train_end_ts].copy()

    if df_train.empty:
        print(f"  [WARN] {parquet_path.name}: 0 rows in TRAIN split (< {train_end_ts})")
        return {}

    arr = df_train[value_col].dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {}

    q1  = float(np.percentile(arr, 25))
    q3  = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mean = float(np.mean(arr))
    std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

    return {
        "file":            parquet_path.name,
        "train_rows_used": int(arr.size),
        "train_end":       train_end_iso,
        "q1":              round(q1,   6),
        "q3":              round(q3,   6),
        "iqr":             round(iqr,  6),
        "mean":            round(mean, 6),
        "std":             round(std,  6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute IQR + normalization stats STRICTLY on the TRAIN split (May 19-22 normal data)."
    )
    ap.add_argument("--clean-dir",  type=Path, default=Path("structural_clean"),
                    help="Dir containing structurally cleaned parquets")
    ap.add_argument("--out",        type=Path, default=Path("artifacts/eda/train_stats.json"),
                    help="Output path for computed stats")
    ap.add_argument("--time-col",   type=str, default="timestamp")
    ap.add_argument("--value-col",  type=str, default="value")
    ap.add_argument("--train-end",  type=str, default=TRAIN_END_ISO,
                    help="Override train end boundary ISO string")
    args = ap.parse_args()

    train_end = args.train_end
    print(f"Computing train stats with boundary: {train_end}")
    print(f"(Only data BEFORE this timestamp is used — no future leakage)\n")

    all_stats: Dict[str, Dict[str, Any]] = {}

    folders = ["cpu_data", "disk_data", "memory_data", "slurm_data"]
    for folder in folders:
        folder_path = args.clean_dir / folder
        if not folder_path.exists():
            print(f"  [SKIP] {folder_path} not found")
            continue

        print(f"Processing {folder}...")
        files = sorted(folder_path.glob("*.parquet"))
        folder_stats: Dict[str, Any] = {}
        for f in files:
            res = compute_for_file(f, train_end, args.time_col, args.value_col)
            if res:
                folder_stats[f.name] = res
                print(f"  {f.name}: {res['train_rows_used']:,} train rows  "
                      f"mean={res['mean']:.4f}  std={res['std']:.4f}")
        all_stats[folder] = folder_stats

    ensure_dir(args.out.parent)
    out_payload = {
        "split_type":        "binary_anomaly_detection",
        "train_end_boundary": train_end,
        "note": (
            "Statistics computed STRICTLY on normal-behaviour data (May 19-22). "
            "Anomalous May 23 data is exclusively in the test split. "
            "No data leakage into normalisation or IQR bounds."
        ),
        "folders": all_stats,
    }
    write_json(args.out, out_payload)
    print(f"\nTrain statistics saved to: {args.out}")


if __name__ == "__main__":
    main()
