"""
eda/apply_stats.py
==================
Phase 2: Apply train-derived IQR + z-score stats to split parquets.

Schema-aware for the two working modalities:
  memory_data : [__name__, instance, job, timestamp, value, hw_config]
                Stats computed per (__name__, hw_config) group from train only.
  slurm_data  : [__name__, instance, job, node, status, timestamp, value, hw_config]
                Stats computed per (__name__, hw_config) group from train only.
                'status' column preserved as-is (non-numeric, not normalised).

Adds columns:
  is_outlier  : bool  — (value < q1 - 3*iqr) or (value > q3 + 3*iqr), from TRAIN stats
  value_norm  : float — z-score, (value - mean) / std, from TRAIN stats
  split       : str   — 'train' | 'test'

Hard constraints:
  - Stats NEVER computed on test data
  - No rows dropped (flag only, never drop)
  - Train outlier rate > 5% raises error and halts

Usage:
  python -m eda.apply_stats --splits-dir artifacts/splits --out-dir cleaned_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from eda.lib import ensure_dir, write_json

TRAIN_OUTLIER_HARD_LIMIT = 0.05   # 5% — raise error if exceeded
IQR_MULTIPLIER_DEFAULT   = 3.0


def compute_train_stats(
    df_train: pd.DataFrame,
    group_cols: list,
    value_col: str,
) -> Dict[Tuple, Dict[str, float]]:
    """
    Compute IQR + z-score stats per group from TRAIN data ONLY.
    Returns dict keyed by tuple of group values.
    """
    stats: Dict[Tuple, Dict[str, float]] = {}

    for keys, grp in df_train.groupby(group_cols, sort=False):
        arr = grp[value_col].dropna().to_numpy(dtype=float)
        if arr.size < 10:
            continue
        key = keys if isinstance(keys, tuple) else (keys,)
        q1  = float(np.percentile(arr, 25))
        q3  = float(np.percentile(arr, 75))
        iqr = q3 - q1
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        stats[key] = {"q1": q1, "q3": q3, "iqr": iqr, "mean": mean, "std": std}

    return stats


def apply_stats_to_df(
    df: pd.DataFrame,
    stats: Dict[Tuple, Dict[str, float]],
    group_cols: list,
    value_col: str,
    split_label: str,
    iqr_multiplier: float,
) -> pd.DataFrame:
    """Apply outlier flag and z-score normalisation in-place."""
    df = df.copy()
    df["split"]      = split_label
    df["is_outlier"] = False
    df["value_norm"] = np.nan

    for keys, grp_idx in df.groupby(group_cols, sort=False).groups.items():
        key = keys if isinstance(keys, tuple) else (keys,)
        st = stats.get(key)
        if st is None:
            continue
        vals = df.loc[grp_idx, value_col]
        lb = st["q1"] - iqr_multiplier * st["iqr"]
        ub = st["q3"] + iqr_multiplier * st["iqr"]
        df.loc[grp_idx, "is_outlier"] = (vals < lb) | (vals > ub)
        if st["std"] > 0:
            df.loc[grp_idx, "value_norm"] = (vals - st["mean"]) / st["std"]
        else:
            df.loc[grp_idx, "value_norm"] = 0.0

    return df


def process_modality(
    modality: str,
    splits_dir: Path,
    out_dir: Path,
    group_cols: list,
    value_col: str,
    iqr_multiplier: float,
) -> Dict[str, Any]:
    """Process a single modality: load train+test, compute stats on train, apply to both."""
    train_path = splits_dir / (modality + "_train.parquet")
    test_path  = splits_dir / (modality + "_test.parquet")

    if not train_path.exists():
        print("  [SKIP] " + str(train_path) + " not found")
        return {}

    print("\n  Loading " + modality + "...")
    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()

    print("  Train rows: " + str(len(df_train)) + "  |  Test rows: " + str(len(df_test)))
    print("  Train columns: " + str(list(df_train.columns)))

    # Compute stats on TRAIN only
    valid_group_cols = [c for c in group_cols if c in df_train.columns]
    print("  Computing stats grouped by: " + str(valid_group_cols))
    stats = compute_train_stats(df_train, valid_group_cols, value_col)
    print("  Stat groups computed: " + str(len(stats)))

    # Apply to train
    df_train_enriched = apply_stats_to_df(df_train, stats, valid_group_cols, value_col, "train", iqr_multiplier)
    train_outlier_rate = df_train_enriched["is_outlier"].mean()
    print("  Train outlier rate: " + str(round(train_outlier_rate * 100, 3)) + "%")

    if train_outlier_rate > TRAIN_OUTLIER_HARD_LIMIT:
        print("\n  [HARD STOP] Train outlier rate " + str(round(train_outlier_rate*100, 2)) +
              "% exceeds 5% limit!")
        print("  IQR multiplier may be too tight. Current: " + str(iqr_multiplier) + "x")
        print("  Halting to prevent corrupted stats.")
        sys.exit(1)

    # Apply to test
    test_outlier_rate = None
    df_test_enriched = pd.DataFrame()
    if not df_test.empty:
        df_test_enriched = apply_stats_to_df(df_test, stats, valid_group_cols, value_col, "test", iqr_multiplier)
        test_outlier_rate = df_test_enriched["is_outlier"].mean()
        print("  Test  outlier rate: " + str(round(test_outlier_rate * 100, 3)) + "%")
        if test_outlier_rate is not None and train_outlier_rate > 0:
            ratio = test_outlier_rate / train_outlier_rate
            print("  Test/Train outlier ratio: " + str(round(ratio, 2)) + "x  " +
                  ("(anomaly signal visible)" if ratio > 1.2 else "(weak signal)"))

    # Verify no rows dropped
    assert len(df_train_enriched) == len(df_train), "Row count changed on train!"
    if not df_test.empty:
        assert len(df_test_enriched) == len(df_test), "Row count changed on test!"
    print("  Zero rows dropped: OK")

    # Write
    ensure_dir(out_dir)
    df_train_enriched.to_parquet(out_dir / (modality + "_train.parquet"), index=False, compression="snappy")
    if not df_test_enriched.empty:
        df_test_enriched.to_parquet(out_dir / (modality + "_test.parquet"), index=False, compression="snappy")
    print("  Written to: " + str(out_dir))

    # Column verification
    print("  Output columns: " + str(list(df_train_enriched.columns)))
    assert "is_outlier" in df_train_enriched.columns, "is_outlier column missing!"
    assert "value_norm" in df_train_enriched.columns, "value_norm column missing!"
    print("  is_outlier and value_norm present: OK")

    return {
        "modality":            modality,
        "group_cols":          valid_group_cols,
        "iqr_multiplier":      iqr_multiplier,
        "n_stat_groups":       len(stats),
        "train_rows":          int(len(df_train_enriched)),
        "test_rows":           int(len(df_test_enriched)) if not df_test_enriched.empty else 0,
        "train_outlier_pct":   round(float(train_outlier_rate) * 100, 3),
        "test_outlier_pct":    round(float(test_outlier_rate) * 100, 3) if test_outlier_rate is not None else None,
        "rows_dropped_train":  0,
        "rows_dropped_test":   0,
        "stats_from_train_only": True,
        "output_columns":      list(df_train_enriched.columns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply train-derived IQR + z-score stats. Schema-aware for memory/slurm."
    )
    ap.add_argument("--splits-dir",     type=Path, default=Path("artifacts/splits"),
                    help="Input: {modality}_train.parquet and {modality}_test.parquet")
    ap.add_argument("--out-dir",        type=Path, default=Path("cleaned_data"),
                    help="Output: enriched parquets with is_outlier + value_norm")
    ap.add_argument("--iqr-multiplier", type=float, default=IQR_MULTIPLIER_DEFAULT,
                    help="IQR multiplier for outlier flagging (default 3.0 i.e. 3-sigma)")
    ap.add_argument("--value-col",      type=str,   default="value")
    args = ap.parse_args()

    # Group by metric name + hw_config for both modalities
    # This ensures stats are computed per metric per hardware class (avoids mixing baselines)
    modality_groups = {
        "memory_data": ["__name__", "hw_config"],
        "slurm_data":  ["__name__", "hw_config"],
    }

    print("=" * 60)
    print("  apply_stats — Train-derived-only IQR + Z-score normalisation")
    print("  IQR multiplier: " + str(args.iqr_multiplier) + "x")
    print("  Hard stop if train outlier rate > " + str(TRAIN_OUTLIER_HARD_LIMIT * 100) + "%")
    print("=" * 60)

    results: Dict[str, Any] = {}
    for modality, group_cols in modality_groups.items():
        result = process_modality(
            modality=modality,
            splits_dir=args.splits_dir,
            out_dir=args.out_dir,
            group_cols=group_cols,
            value_col=args.value_col,
            iqr_multiplier=args.iqr_multiplier,
        )
        if result:
            results[modality] = result

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print("  Modality         Train OutlierPct   Test OutlierPct   Rows Dropped")
    print("  " + "-" * 65)
    for mod, r in results.items():
        train_pct = str(r["train_outlier_pct"]) + "%"
        test_pct  = str(r["test_outlier_pct"])  + "%" if r["test_outlier_pct"] is not None else "N/A"
        dropped   = r["rows_dropped_train"] + r["rows_dropped_test"]
        print("  " + mod.ljust(16) + train_pct.rjust(18) + test_pct.rjust(18) + str(dropped).rjust(14))

    # Write versioning manifest
    manifest = {
        "generator":       "eda.apply_stats",
        "iqr_multiplier":  args.iqr_multiplier,
        "train_only_stats": True,
        "modalities":      results,
        "train_boundary":  "2023-05-22T23:59:59 (normal data only)",
        "test_boundary":   "2023-05-23T00:00:00 -> 2023-05-23T23:59:59 (anomalous event)",
        "note":            "is_outlier uses train-derived IQR bounds. value_norm uses train mean/std.",
    }
    ensure_dir(args.out_dir)
    write_json(args.out_dir / "dataset_versioning.json", manifest)
    print("\n  Manifest written: " + str(args.out_dir / "dataset_versioning.json"))


if __name__ == "__main__":
    main()
