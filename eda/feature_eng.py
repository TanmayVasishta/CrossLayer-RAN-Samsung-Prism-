"""
eda/feature_eng.py
==================
Feature engineering for unsupervised anomaly detection.
Operates on the binary split parquets from artifacts/splits/.

Only two modalities have May 23 coverage:
  memory_data : [__name__, instance, hw_config, timestamp, value]
                47 metrics per node (332 nodes), absolute bytes, scraped 1/min.
                Strategy: rolling windows (5/15/60min) + lag-1 diff per (instance, __name__)
                          then pivot to wide format: one row per (instance, timestamp)

  slurm_data  : [__name__, instance, job, node, status, hw_config, timestamp, value]
                4 metrics per node (alloc/idle/other/total CPUs), scraped every 30s.
                Strategy: pivot __name__ to columns -> compute ratios -> encode status
                          -> is_distress_status binary flag -> status transition rate

Outputs:
  artifacts/features/memory_data_train_features.parquet
  artifacts/features/memory_data_test_features.parquet
  artifacts/features/slurm_data_train_features.parquet
  artifacts/features/slurm_data_test_features.parquet
  artifacts/features/feature_manifest.json

Hard constraints:
  - Rolling window parameters fitted on TRAIN data shape only (no lookahead into test)
  - No scaler fitted on test data
  - Rows with >30% NaN features after rolling warmup are dropped (documented in manifest)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.lib import ensure_dir, write_json

# ─── Constants ─────────────────────────────────────────────────────────────────
DISTRESS_STATES = frozenset({
    "down", "drained", "draining", "fail", "failing",
    "down*", "drained*", "drain", "failing*",
})

# Rolling window sizes for memory (as time offsets, matching 1T resample)
MEMORY_WINDOWS = ["5min", "15min", "60min"]
MEMORY_AGG     = ["mean", "std", "min", "max"]

# Max fraction of NaN features allowed before dropping a row
NAN_DROP_THRESHOLD = 0.30


# ─── Memory feature engineering ────────────────────────────────────────────────

def build_memory_features(df: pd.DataFrame, out_path: Path, is_train: bool) -> tuple[int, list[str]]:
    """
    Build rolling window + lag-diff features for memory_data.
    Processes chunk-by-chunk (per instance) and streams directly to Parquet to save RAM.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    time_col     = "timestamp"
    value_col    = "value"
    instance_col = "instance"
    name_col     = "__name__"

    if df.empty:
        return 0, []

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    tag = "TRAIN" if is_train else "TEST"
    print(f"  [{tag}] Unique instances: {df[instance_col].nunique()}, metrics: {df[name_col].nunique()}")

    writer = None
    total_rows = 0
    total_dropped = 0
    feature_names = []

    for inst, inst_df in df.groupby(instance_col, sort=False):
        pieces = []
        hw = inst_df["hw_config"].iloc[0]

        for metric, grp in inst_df.groupby(name_col, sort=False):
            g = grp.set_index(time_col)[[value_col]].sort_index()

            # Safe feature names (replace non-alphanum with _)
            metric_safe = metric.replace("node_memory_", "").replace("/", "_").replace("-", "_")

            feat_cols: Dict[str, pd.Series] = {}

            # Rolling window features
            for window in MEMORY_WINDOWS:
                roller = g[value_col].rolling(window=window, min_periods=1)
                w_safe = window.replace("min", "m")
                for agg in MEMORY_AGG:
                    col_name = f"{metric_safe}_{w_safe}_{agg}"
                    if agg == "mean":
                        feat_cols[col_name] = roller.mean()
                    elif agg == "std":
                        feat_cols[col_name] = roller.std().fillna(0.0)
                    elif agg == "min":
                        feat_cols[col_name] = roller.min()
                    elif agg == "max":
                        feat_cols[col_name] = roller.max()

            # Lag-1 difference
            feat_cols[f"{metric_safe}_lag1_diff"] = g[value_col].diff(1)

            pieces.append(pd.DataFrame(feat_cols, index=g.index))

        if not pieces:
            continue

        # Join all metrics for this instance on the time index directly
        inst_wide = pd.concat(pieces, axis=1).reset_index()
        inst_wide[instance_col] = inst
        inst_wide["hw_config"]  = hw

        # Drop rows with >30% NaN features
        feat_cols_final = [c for c in inst_wide.columns if c not in [instance_col, time_col, "hw_config"]]
        n_before = len(inst_wide)
        nan_frac  = inst_wide[feat_cols_final].isna().mean(axis=1)
        inst_wide = inst_wide[nan_frac <= NAN_DROP_THRESHOLD].reset_index(drop=True)
        
        total_dropped += (n_before - len(inst_wide))
        total_rows += len(inst_wide)
        
        if inst_wide.empty:
            continue
            
        if not feature_names:
            feature_names = feat_cols_final

        # Stream chunk to parquet
        table = pa.Table.from_pandas(inst_wide, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"  [{tag}] Wide shape: ({total_rows}, {len(feature_names) + 3})  |  NaN-dropped rows: {total_dropped:,}")
    return total_rows, feature_names


# ─── Slurm feature engineering ─────────────────────────────────────────────────

def build_slurm_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Build ratio + status-encoding features for slurm_data.

    Input columns: [__name__, instance, job, node, status, hw_config, timestamp, value]
    Output: one row per (node, timestamp) with:
      cpu_alloc_ratio, cpu_idle_ratio, cpu_other_ratio
      status_* one-hot columns
      is_distress_status (critical binary flag)
      status_transition_rate_15min
      hw_config
    """
    time_col   = "timestamp"
    value_col  = "value"
    name_col   = "__name__"
    node_col   = "node"
    status_col = "status"

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    tag = "TRAIN" if is_train else "TEST"
    print(f"  [{tag}] Unique nodes: {df[node_col].nunique()}, metrics: {df[name_col].unique().tolist()}")

    # ── Step 1: Pivot __name__ to columns ──────────────────────────────────────
    # Each (node, timestamp, status) row should have alloc + idle values
    # status can differ between rows at same timestamp — use most common status
    pivot_val = df.pivot_table(
        index=[node_col, time_col, status_col, "hw_config"],
        columns=name_col,
        values=value_col,
        aggfunc="first",
    ).reset_index()

    # Rename metric columns to clean names
    col_rename = {}
    for c in pivot_val.columns:
        if c.startswith("slurm_node_cpu_"):
            col_rename[c] = c.replace("slurm_node_cpu_", "cpu_")
    pivot_val = pivot_val.rename(columns=col_rename)

    # ── Step 2: CPU utilisation ratios ─────────────────────────────────────────
    # alloc/total, idle/total, other/total — more informative than raw counts
    for ratio_col in ["cpu_alloc", "cpu_idle", "cpu_other"]:
        if ratio_col in pivot_val.columns and "cpu_total" in pivot_val.columns:
            pivot_val[ratio_col + "_ratio"] = np.where(
                pivot_val["cpu_total"] > 0,
                pivot_val[ratio_col] / pivot_val["cpu_total"],
                np.nan,
            )

    # ── Step 3: Status encoding ────────────────────────────────────────────────
    # is_distress_status: primary anomaly proxy signal
    pivot_val["is_distress_status"] = (
        pivot_val[status_col].str.lower()
        .isin(DISTRESS_STATES)
        .astype(int)
    )

    # One-hot encode status (top N states to avoid explosion)
    top_statuses = df[status_col].value_counts().head(10).index.tolist()
    for s in top_statuses:
        safe_name = "status_" + s.replace("*", "_star").replace("-", "_").replace(" ", "_")
        pivot_val[safe_name] = (pivot_val[status_col] == s).astype(int)

    # ── Step 4: Status transition rate (per node, 15min rolling window) ────────
    # Captures nodes that flip between states — a strong anomaly signal
    pieces: List[pd.DataFrame] = []
    for node_id, node_grp in pivot_val.groupby(node_col, sort=False):
        node_grp = node_grp.sort_values(time_col)
        # Encode status as integer for diff calculation
        status_int = pd.Categorical(node_grp[status_col]).codes
        status_series = pd.Series(status_int, index=node_grp[time_col])
        # Count status changes in rolling 15min window
        changes = (status_series.diff().abs() > 0).astype(int)
        try:
            transition_rate = changes.rolling("15min", min_periods=1).sum()
        except Exception:
            transition_rate = pd.Series(0, index=changes.index)
        node_grp = node_grp.copy()
        node_grp["status_transition_rate_15min"] = transition_rate.values
        pieces.append(node_grp)

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, ignore_index=True)

    # ── Step 5: Drop rows with >30% NaN features ───────────────────────────────
    meta_cols = [node_col, time_col, status_col, "hw_config", "is_distress_status"]
    feat_cols = [c for c in result.columns if c not in meta_cols]
    n_before = len(result)
    nan_frac  = result[feat_cols].isna().mean(axis=1)
    result    = result[nan_frac <= NAN_DROP_THRESHOLD].reset_index(drop=True)
    n_dropped = n_before - len(result)

    print(f"  [{tag}] Output shape: {result.shape}  |  NaN-dropped rows: {n_dropped:,}")
    distress_pct = result["is_distress_status"].mean() * 100
    print(f"  [{tag}] is_distress_status rate: {distress_pct:.3f}%")

    return result


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Feature engineering for memory_data and slurm_data anomaly detection."
    )
    ap.add_argument("--splits-dir", type=Path, default=Path("artifacts/splits"),
                    help="Input split parquets directory")
    ap.add_argument("--out-dir",    type=Path, default=Path("artifacts/features"),
                    help="Output directory for feature parquets")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest: Dict[str, Any] = {
        "generated_at":     __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "nan_drop_threshold": NAN_DROP_THRESHOLD,
        "memory_windows":   MEMORY_WINDOWS,
        "memory_agg_funcs": MEMORY_AGG,
        "slurm_status_transition_window": "15min",
        "distress_states":  sorted(DISTRESS_STATES),
        "modalities":       {},
    }

    # ─── memory_data ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Feature Engineering: memory_data")
    print("=" * 60)
    mem_train_path = args.splits_dir / "memory_data_train.parquet"
    mem_test_path  = args.splits_dir / "memory_data_test.parquet"

    if mem_train_path.exists():
        df_mem_train = pd.read_parquet(mem_train_path)
        df_mem_test  = pd.read_parquet(mem_test_path) if mem_test_path.exists() else pd.DataFrame()

        print(f"  Train input shape: {df_mem_train.shape}")
        print(f"  Test  input shape: {df_mem_test.shape}")

        out_train_path = args.out_dir / "memory_data_train_features.parquet"
        out_test_path  = args.out_dir / "memory_data_test_features.parquet"

        train_rows, train_feat_names = build_memory_features(df_mem_train, out_train_path, is_train=True)
        test_rows, test_feat_names = build_memory_features(df_mem_test, out_test_path, is_train=False)

        feat_names = train_feat_names if train_feat_names else test_feat_names
        
        manifest["modalities"]["memory_data"] = {
            "train_input_shape":    list(df_mem_train.shape),
            "test_input_shape":     list(df_mem_test.shape) if not df_mem_test.empty else [0, 0],
            "train_output_shape":   [train_rows, len(feat_names) + 3],
            "test_output_shape":    [test_rows, len(feat_names) + 3],
            "n_features":           len(feat_names),
            "feature_names":        feat_names,
            "rolling_windows":      MEMORY_WINDOWS,
            "agg_funcs":            MEMORY_AGG,
            "lag_diff_included":    True,
        }
        print(f"\n  memory_data train features: {train_rows} rows")
        print(f"  memory_data test  features: {test_rows} rows")
    else:
        print("  [SKIP] memory_data_train.parquet not found in " + str(args.splits_dir))

    # ─── slurm_data ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Feature Engineering: slurm_data")
    print("=" * 60)
    slurm_train_path = args.splits_dir / "slurm_data_train.parquet"
    slurm_test_path  = args.splits_dir / "slurm_data_test.parquet"

    if slurm_train_path.exists():
        df_slurm_train = pd.read_parquet(slurm_train_path)
        df_slurm_test  = pd.read_parquet(slurm_test_path) if slurm_test_path.exists() else pd.DataFrame()

        print(f"  Train input shape: {df_slurm_train.shape}")
        print(f"  Test  input shape: {df_slurm_test.shape}")

        feats_slurm_train = build_slurm_features(df_slurm_train, is_train=True)
        feats_slurm_test  = build_slurm_features(df_slurm_test,  is_train=False) if not df_slurm_test.empty else pd.DataFrame()

        out_train_path = args.out_dir / "slurm_data_train_features.parquet"
        out_test_path  = args.out_dir / "slurm_data_test_features.parquet"
        feats_slurm_train.to_parquet(out_train_path, index=False, compression="snappy")
        if not feats_slurm_test.empty:
            feats_slurm_test.to_parquet(out_test_path, index=False, compression="snappy")

        feat_names = [c for c in feats_slurm_train.columns if c not in ["node", "timestamp", "status", "hw_config"]]
        manifest["modalities"]["slurm_data"] = {
            "train_input_shape":    list(df_slurm_train.shape),
            "test_input_shape":     list(df_slurm_test.shape) if not df_slurm_test.empty else [0, 0],
            "train_output_shape":   list(feats_slurm_train.shape),
            "test_output_shape":    list(feats_slurm_test.shape) if not feats_slurm_test.empty else [0, 0],
            "n_features":           len(feat_names),
            "feature_names":        feat_names,
            "is_distress_status_included": True,
            "distress_states":      sorted(DISTRESS_STATES),
        }
        print(f"\n  slurm_data train features: {feats_slurm_train.shape}")
        print(f"  slurm_data test  features: {feats_slurm_test.shape if not feats_slurm_test.empty else '(empty)'}")
    else:
        print("  [SKIP] slurm_data_train.parquet not found in " + str(args.splits_dir))

    # Write manifest
    write_json(args.out_dir / "feature_manifest.json", manifest)
    print("\n  Feature manifest: " + str(args.out_dir / "feature_manifest.json"))


if __name__ == "__main__":
    main()
