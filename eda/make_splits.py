"""
eda/make_splits.py
==================
Correct chronological splitting for UNSUPERVISED ANOMALY DETECTION.

Split design:
  TRAIN : May 19 00:00 -> May 22 23:59  (normal cluster behaviour ONLY)
  TEST  : May 23 00:00 -> May 23 23:59  (real anomalous JLab event)
  VAL   : NONE — unsupervised, no labels

Schema facts (from paper):
  memory_data : columns = [__name__, instance, job, timestamp, value]
                __name__ = metric name (e.g. node_memory_Active_bytes)
                instance = node identifier (e.g. farm140151:9100)
                hw_config = instance[:6]  ->  farm14 / farm16 / farm18 / farm19 / farm23
                Values are absolute bytes (NOT counters). Scraped ~1/min.

  slurm_data  : columns = [__name__, instance, job, node, status, timestamp, value]
                __name__ = one of 4 metrics: alloc/idle/other/total CPU counts
                node = actual node name (e.g. farm140105)
                hw_config = node[:6]  ->  farm14 / farm16 / farm18 / farm19 / farm23
                status = node state: allocated/idle/mixed/down/drained/draining/fail/failing...
                Values are CPU counts. Scraped every 30s. NO resampling applied.

  cpu_data    : cumulative counters — EXCLUDED (zero May 23 coverage)
  disk_data   : cumulative counters — EXCLUDED (zero May 23 coverage)

Per-modality resample:
  memory_data  : 1T (1-min), grouped by (instance, __name__) — preserves metric identity
  slurm_data   : None — event-driven, no resampling
  cpu_data     : excluded
  disk_data    : excluded

Outputs:
  artifacts/splits/{modality}_train.parquet
  artifacts/splits/{modality}_test.parquet
  artifacts/splits/split_manifest.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.lib import ensure_dir, write_json

# ---------------------------------------------------------------------------
# Split boundaries
# ---------------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2023-05-19 00:00:00")
TRAIN_END   = pd.Timestamp("2023-05-22 23:59:59.999999999")
TEST_START  = pd.Timestamp("2023-05-23 00:00:00")
TEST_END    = pd.Timestamp("2023-05-23 23:59:59.999999999")

# Hardware config IDs from the paper
HW_CONFIGS = {"farm14", "farm16", "farm18", "farm19", "farm23"}
HW_CONFIG_FALLBACK = "unknown"

MAX_FFILL_STEPS = 3


def extract_hw_config(node_str: str) -> str:
    """Extract hw_config from first 6 chars of instance/node string."""
    clean = str(node_str).split(":")[0][:6].lower()
    return clean if clean in HW_CONFIGS else HW_CONFIG_FALLBACK


# ---------------------------------------------------------------------------
# Memory data splitting (1-min resample, grouped by instance + metric name)
# ---------------------------------------------------------------------------
def process_memory(
    folder_dir: Path,
    out_dir: Path,
    time_col: str = "timestamp",
    value_col: str = "value",
    name_col: str = "__name__",
    instance_col: str = "instance",
) -> Dict[str, Any]:
    files = sorted(folder_dir.glob("*.parquet"))
    if not files:
        return {}

    print("\n" + "=" * 60)
    print("  Modality: memory_data  |  files: " + str(len(files)) + "  |  resample: 1T")
    print("=" * 60)

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df_f = pd.read_parquet(f)
            frames.append(df_f)
            print("    Read " + f.name + ": " + str(len(df_f)) + " rows")
        except Exception as e:
            print("    [ERROR] " + f.name + ": " + str(e))
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    total_rows = len(df)
    print("  Combined: " + str(total_rows) + " rows")

    # hw_config from first 6 chars of instance (before the colon+port)
    df["hw_config"] = df[instance_col].apply(extract_hw_config)
    print("  hw_config distribution:")
    print(df["hw_config"].value_counts().to_string())

    # Binary split
    train_mask = (df[time_col] >= TRAIN_START) & (df[time_col] <= TRAIN_END)
    test_mask  = (df[time_col] >= TEST_START)  & (df[time_col] <= TEST_END)
    df_train_raw = df[train_mask].copy()
    df_test_raw  = df[test_mask].copy()

    print("  Raw train: " + str(len(df_train_raw)) + "  |  Raw test: " + str(len(df_test_raw)))

    def resample_memory(df_in: pd.DataFrame, tag: str) -> pd.DataFrame:
        """Resample memory data at 1-min, grouped by (instance, __name__)."""
        if df_in.empty:
            return df_in
        pieces = []
        grouped = df_in.groupby([instance_col, name_col], sort=False)
        for (inst, metric), grp in grouped:
            hw = grp["hw_config"].iloc[0]
            g = grp.set_index(time_col)[[value_col]].sort_index()
            resampled = g.resample("1min").mean()
            resampled = resampled.ffill(limit=MAX_FFILL_STEPS).dropna()
            resampled[instance_col] = inst
            resampled[name_col]     = metric
            resampled["hw_config"]  = hw
            resampled = resampled.reset_index()
            pieces.append(resampled)
        if not pieces:
            return pd.DataFrame()
        out = pd.concat(pieces, ignore_index=True)
        print("  [" + tag + "] After 1T resample: " + str(len(df_in)) + " -> " + str(len(out)) + " rows")
        return out

    df_train = resample_memory(df_train_raw, "TRAIN")
    df_test  = resample_memory(df_test_raw,  "TEST")

    # Verify zero overlap
    zero_overlap = None
    if len(df_train) and len(df_test):
        max_train = df_train[time_col].max()
        min_test  = df_test[time_col].min()
        zero_overlap = bool(max_train < min_test)
        print("  max_train=" + str(max_train) + "  min_test=" + str(min_test))
        print("  Zero overlap: " + ("OK" if zero_overlap else "FAIL"))

    # Write
    ensure_dir(out_dir)
    df_train.to_parquet(out_dir / "memory_data_train.parquet", index=False, compression="snappy")
    df_test.to_parquet(out_dir  / "memory_data_test.parquet",  index=False, compression="snappy")

    n_train, n_test = len(df_train), len(df_test)
    n_total = n_train + n_test
    print("\n  --- Sanity check: memory_data ---")
    print("    train shape: " + str(df_train.shape) + "  (" + str(round(100*n_train/n_total, 1) if n_total else 0) + "%)")
    print("    test  shape: " + str(df_test.shape)  + "  (" + str(round(100*n_test/n_total, 1) if n_total else 0)  + "%)")
    if len(df_train):
        print("    train unique instances: " + str(df_train[instance_col].nunique()))
        print("    train unique metrics:   " + str(df_train[name_col].nunique()))

    return {
        "modality": "memory_data", "resample_freq": "1min", "ffill_limit": MAX_FFILL_STEPS,
        "source_files": [f.name for f in files], "raw_total_rows": int(total_rows),
        "train": {"rows": int(n_train), "shape": list(df_train.shape),
                  "pct_of_eligible": round(100*n_train/n_total, 2) if n_total else 0,
                  "time_start": str(df_train[time_col].min()) if n_train else None,
                  "time_end":   str(df_train[time_col].max()) if n_train else None,
                  "unique_instances": int(df_train[instance_col].nunique()) if n_train else 0,
                  "unique_metrics":   int(df_train[name_col].nunique()) if n_train else 0,
                  "hw_config_counts": df_train["hw_config"].value_counts().to_dict() if n_train else {}},
        "test":  {"rows": int(n_test),  "shape": list(df_test.shape),
                  "pct_of_eligible": round(100*n_test/n_total, 2) if n_total else 0,
                  "time_start": str(df_test[time_col].min()) if n_test else None,
                  "time_end":   str(df_test[time_col].max()) if n_test else None,
                  "unique_instances": int(df_test[instance_col].nunique()) if n_test else 0,
                  "unique_metrics":   int(df_test[name_col].nunique()) if n_test else 0,
                  "hw_config_counts": df_test["hw_config"].value_counts().to_dict() if n_test else {}},
        "zero_temporal_overlap": zero_overlap,
    }


# ---------------------------------------------------------------------------
# Slurm data splitting (no resampling — event-driven)
# ---------------------------------------------------------------------------
def process_slurm(
    folder_dir: Path,
    out_dir: Path,
    time_col: str = "timestamp",
    value_col: str = "value",
    name_col: str = "__name__",
    node_col: str = "node",
    status_col: str = "status",
) -> Dict[str, Any]:
    files = sorted(folder_dir.glob("*.parquet"))
    if not files:
        return {}

    print("\n" + "=" * 60)
    print("  Modality: slurm_data  |  files: " + str(len(files)) + "  |  resample: None (event-driven)")
    print("=" * 60)

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df_f = pd.read_parquet(f)
            frames.append(df_f)
            print("    Read " + f.name + ": " + str(len(df_f)) + " rows")
        except Exception as e:
            print("    [ERROR] " + f.name + ": " + str(e))
    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    total_rows = len(df)
    print("  Combined: " + str(total_rows) + " rows")

    # hw_config from first 6 chars of node column
    df["hw_config"] = df[node_col].apply(extract_hw_config)
    print("  hw_config distribution:")
    print(df["hw_config"].value_counts().to_string())

    # Status column summary
    print("  status distribution (top 10):")
    print(df[status_col].value_counts().head(10).to_string())

    # Binary split
    train_mask = (df[time_col] >= TRAIN_START) & (df[time_col] <= TRAIN_END)
    test_mask  = (df[time_col] >= TEST_START)  & (df[time_col] <= TEST_END)
    df_train = df[train_mask].sort_values(time_col).reset_index(drop=True)
    df_test  = df[test_mask].sort_values(time_col).reset_index(drop=True)

    print("  Raw train: " + str(len(df_train)) + "  |  Raw test: " + str(len(df_test)))

    # Verify zero overlap
    zero_overlap = None
    if len(df_train) and len(df_test):
        max_train = df_train[time_col].max()
        min_test  = df_test[time_col].min()
        zero_overlap = bool(max_train < min_test)
        print("  max_train=" + str(max_train) + "  min_test=" + str(min_test))
        print("  Zero overlap: " + ("OK" if zero_overlap else "FAIL"))

    # Show distress status in test split
    if len(df_test):
        distress_states = {"down", "drained", "draining", "fail", "failing", "down*", "drained*"}
        test_distress = df_test[df_test[status_col].str.lower().isin(distress_states)]
        print("  May 23 distress-state rows: " + str(len(test_distress)) +
              " (" + str(round(100*len(test_distress)/len(df_test), 2)) + "% of test)")
        print("  Distress status breakdown:")
        print(test_distress[status_col].value_counts().to_string())

    # Write
    ensure_dir(out_dir)
    df_train.to_parquet(out_dir / "slurm_data_train.parquet", index=False, compression="snappy")
    df_test.to_parquet(out_dir  / "slurm_data_test.parquet",  index=False, compression="snappy")

    n_train, n_test = len(df_train), len(df_test)
    n_total = n_train + n_test
    print("\n  --- Sanity check: slurm_data ---")
    print("    train shape: " + str(df_train.shape) + "  (" + str(round(100*n_train/n_total, 1) if n_total else 0) + "%)")
    print("    test  shape: " + str(df_test.shape)  + "  (" + str(round(100*n_test/n_total, 1) if n_total else 0)  + "%)")
    print("    train unique nodes: " + str(df_train[node_col].nunique()))
    print("    test  unique nodes: " + str(df_test[node_col].nunique()))
    print("    unique __name__: " + str(df[name_col].unique().tolist()))

    return {
        "modality": "slurm_data", "resample_freq": "none", "ffill_limit": 0,
        "source_files": [f.name for f in files], "raw_total_rows": int(total_rows),
        "train": {"rows": int(n_train), "shape": list(df_train.shape),
                  "pct_of_eligible": round(100*n_train/n_total, 2) if n_total else 0,
                  "time_start": str(df_train[time_col].min()) if n_train else None,
                  "time_end":   str(df_train[time_col].max()) if n_train else None,
                  "unique_nodes": int(df_train[node_col].nunique()),
                  "hw_config_counts": df_train["hw_config"].value_counts().to_dict()},
        "test":  {"rows": int(n_test),  "shape": list(df_test.shape),
                  "pct_of_eligible": round(100*n_test/n_total, 2) if n_total else 0,
                  "time_start": str(df_test[time_col].min()) if n_test else None,
                  "time_end":   str(df_test[time_col].max()) if n_test else None,
                  "unique_nodes": int(df_test[node_col].nunique()),
                  "hw_config_counts": df_test["hw_config"].value_counts().to_dict()},
        "zero_temporal_overlap": zero_overlap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Anomaly-detection-aware chronological splitter.\n"
            "Train=May19-22 (normal), Test=May23 (anomalous).\n"
            "Only memory_data and slurm_data processed — cpu/disk excluded (zero May23 coverage)."
        )
    )
    ap.add_argument("--clean-dir", type=Path, default=Path("structural_clean"))
    ap.add_argument("--out-dir",   type=Path, default=Path("artifacts/splits"))
    ap.add_argument("--time-col",  type=str,  default="timestamp")
    ap.add_argument("--value-col", type=str,  default="value")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_policy": {
            "type":      "binary_chronological_anomaly_detection",
            "train":     "2023-05-19 00:00 -> 2023-05-22 23:59  (normal behaviour)",
            "test":      "2023-05-23 00:00 -> 2023-05-23 23:59  (anomalous JLab event)",
            "val":       "NONE — unsupervised pipeline, no labels",
            "constraint": "Train MUST contain ONLY normal data. All anomaly signal is in test.",
        },
        "exclusions": {
            "cpu_data":  "EXCLUDED — zero May23 coverage (84% gap in Prometheus scraping)",
            "disk_data": "EXCLUDED — zero May23 coverage",
        },
        "hw_config_key": "first 6 chars of instance[:6] (memory) or node[:6] (slurm)",
        "hw_configs":    sorted(HW_CONFIGS),
        "modalities":   {},
    }

    # Process memory_data
    mem_dir = args.clean_dir / "memory_data"
    if mem_dir.exists():
        entry = process_memory(mem_dir, args.out_dir, args.time_col, args.value_col)
        if entry:
            manifest["modalities"]["memory_data"] = entry
    else:
        print("[SKIP] " + str(mem_dir) + " not found")

    # Process slurm_data
    slurm_dir = args.clean_dir / "slurm_data"
    if slurm_dir.exists():
        entry = process_slurm(slurm_dir, args.out_dir, args.time_col, args.value_col)
        if entry:
            manifest["modalities"]["slurm_data"] = entry
    else:
        print("[SKIP] " + str(slurm_dir) + " not found")

    manifest_path = args.out_dir / "split_manifest.json"
    write_json(manifest_path, manifest)
    print("\n" + "=" * 60)
    print("  Split manifest saved: " + str(manifest_path))
    print("=" * 60)

    print("\n  Modality         Train rows    Test rows  Resample  Overlap-free")
    print("  " + "-" * 65)
    for mod, entry in manifest["modalities"].items():
        tr = entry["train"]["rows"]
        te = entry["test"]["rows"]
        rs = entry["resample_freq"]
        ov = entry.get("zero_temporal_overlap")
        ov_str = "OK" if ov else ("N/A" if ov is None else "FAIL")
        print("  " + mod.ljust(16) + str(tr).rjust(12) + str(te).rjust(12) + rs.rjust(10) + ov_str.rjust(14))


if __name__ == "__main__":
    main()
