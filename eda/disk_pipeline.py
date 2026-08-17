"""
eda/disk_pipeline.py
====================
Disk I/O anomaly detection pipeline using Polars (mirrors cpu_pipeline_v3.py).

Disk metrics split into two types:
  GAUGE   : node_disk_io_now                     (instantaneous — use directly)
  COUNTER : all *_total and *_seconds_total       (cumulative — need diff for rate)

Strategy per file:
  1. Aggregate across devices (sum per node per timestamp) — keep total disk I/O
  2. Diff counters → rates; keep gauges as-is
  3. Resample to 1-min via Polars group_by_dynamic
  4. Pivot metric names wide per node per timestamp
  5. Compute derived ratios (read/write balance, utilisation)
  6. Write train (May19-22) / test (May23) splits incrementally

Steps:
  1 - Splits   → artifacts/splits/disk_data_train/test.parquet
  2 - Features → artifacts/features/disk_data_train/test_features.parquet
  3 - Training → artifacts/models/disk_isoforest/autoencoder_farmXX.joblib
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models.autoencoder import ReconAE

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START = datetime(2023, 5, 19,  0,  0,  0)
TRAIN_END   = datetime(2023, 5, 22, 23, 59, 59)
TEST_START  = datetime(2023, 5, 23,  0,  0,  0)
TEST_END    = datetime(2023, 5, 23, 23, 59, 59)

HW_CONFIGS   = {"farm14", "farm16", "farm18", "farm19", "farm23"}
RESAMPLE_S   = "1m"

# Gauge metrics (use value directly, no diff)
GAUGE_METRICS = {"node_disk_io_now"}

# Counter metrics → compute rate (diff / dt)
COUNTER_METRICS = {
    "node_disk_io_time_seconds_total",
    "node_disk_io_time_weighted_seconds_total",
    "node_disk_read_bytes_total",
    "node_disk_read_time_seconds_total",
    "node_disk_reads_completed_total",
    "node_disk_reads_merged_total",
    "node_disk_write_time_seconds_total",
    "node_disk_writes_completed_total",
    "node_disk_writes_merged_total",
    "node_disk_written_bytes_total",
}

ALL_METRICS   = GAUGE_METRICS | COUNTER_METRICS
ROLLING_WINS  = [5, 15, 60]          # minutes (using row-count rolling)
NAN_THRESH    = 0.30
IF_CONTAM     = 0.05
IF_LIMIT      = 0.15
PCA_VAR       = 0.95
MIN_ROWS      = 200
AE_SIGMA      = 3.0
FLUSH_EVERY   = 11                    # files per flush (44 files → 4 flushes)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def hw_config_from_instance(inst: str) -> str:
    c = str(inst).split(":")[0][:6].lower()
    return c if c in HW_CONFIGS else "unknown"


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def process_disk_file(f: Path) -> Optional[pl.DataFrame]:
    """
    Load one disk parquet, aggregate across devices, compute rates for counters,
    resample to 1-min, return wide DataFrame per (timestamp, instance, hw_config).
    """
    try:
        lf = pl.scan_parquet(str(f))
    except Exception as e:
        print(f"    [WARN] {f.name}: {e}")
        return None

    # Early filter: time range + known metrics
    lf = (lf
          .filter(pl.col("timestamp") >= pl.lit(TRAIN_START))
          .filter(pl.col("timestamp") <= pl.lit(TEST_END))
          .filter(pl.col("__name__").is_in(list(ALL_METRICS)))
    )

    try:
        df = lf.select(["__name__", "instance", "timestamp", "value"]).collect()
    except Exception as e:
        print(f"    [WARN] collect {f.name}: {e}")
        return None

    if df.is_empty():
        return None

    df = df.with_columns(
        pl.col("instance").map_elements(hw_config_from_instance, return_dtype=pl.String)
          .alias("hw_config")
    ).filter(pl.col("hw_config") != "unknown")

    if df.is_empty():
        return None

    # Determine metric type from file content
    metric_name = df["__name__"][0]
    is_counter   = metric_name in COUNTER_METRICS

    # Aggregate across devices: sum per (instance, hw_config, __name__, timestamp)
    df = (df
          .group_by(["instance", "hw_config", "__name__", "timestamp"])
          .agg(pl.sum("value"))
          .sort(["instance", "__name__", "timestamp"])
    )

    if is_counter:
        # Vectorized diff for rate
        df = df.with_columns([
            pl.col("value").diff().over(["instance", "__name__"]).alias("dv"),
            pl.col("timestamp").diff().dt.total_seconds()
              .over(["instance", "__name__"]).alias("dt_sec"),
        ])
        df = (df
              .with_columns(
                  (pl.col("dv") / pl.col("dt_sec")).clip(lower_bound=0.0).alias("rate")
              )
              .drop_nulls("rate")
              .filter(pl.col("dt_sec") > 0)
              .drop(["value", "dv", "dt_sec"])
              .rename({"rate": "value"})
        )
    # For gauges, value is used directly

    if df.is_empty():
        return None

    # Resample to 1-min
    df = df.sort(["instance", "hw_config", "__name__", "timestamp"])
    rs = (df
          .group_by_dynamic(
              "timestamp",
              every=RESAMPLE_S,
              closed="left",
              group_by=["instance", "hw_config", "__name__"],
          )
          .agg(pl.mean("value"))
    )

    if rs.is_empty():
        return None

    # Pivot metric names wide
    col_name = metric_name + ("_rate" if is_counter else "_now")
    wide = (rs
            .rename({"value": col_name})
            .drop("__name__")
    )

    return wide


def _append_polars_to_parquet(df_pl: pl.DataFrame, path: Path) -> None:
    """Append Polars DataFrame to parquet (read-merge-write pattern)."""
    df_pd = df_pl.to_pandas()
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df_pd], ignore_index=True)
        combined.to_parquet(path, index=False, compression="snappy")
    else:
        df_pd.to_parquet(path, index=False, compression="snappy")


def build_disk_splits(clean_dir: Path, out_dir: Path) -> Dict[str, Any]:
    disk_dir = clean_dir / "disk_data"
    files    = sorted(disk_dir.glob("*.parquet"))
    if not files:
        print(f"[ERROR] No parquets in {disk_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  STEP 1 — DISK SPLITS  |  {len(files)} files  |  {len(ALL_METRICS)} metrics")
    print(f"{'='*60}")

    ensure_dir(out_dir)
    train_path = out_dir / "disk_data_train.parquet"
    test_path  = out_dir / "disk_data_test.parquet"
    train_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    # ── Load ALL disk files in one lazy scan ──────────────────────────────────
    print("  Loading all 44 files via Polars lazy scan...")
    lf = (pl.scan_parquet([str(f) for f in files])
            .filter(pl.col("timestamp") >= pl.lit(TRAIN_START))
            .filter(pl.col("timestamp") <= pl.lit(TEST_END))
            .filter(pl.col("__name__").is_in(list(ALL_METRICS)))
            .select(["__name__", "instance", "timestamp", "value"])
    )
    df = lf.collect()
    print(f"  Collected: {df.shape}")

    if df.is_empty():
        print("  [ERROR] No data after filter"); return {}

    # Add hw_config, filter unknowns
    df = df.with_columns(
        pl.col("instance").map_elements(hw_config_from_instance, return_dtype=pl.String)
          .alias("hw_config")
    ).filter(pl.col("hw_config") != "unknown")

    # ── Aggregate across devices per (instance, __name__, timestamp) ──────────
    print("  Aggregating across devices...")
    df = (df
          .group_by(["instance", "hw_config", "__name__", "timestamp"])
          .agg(pl.sum("value"))
          .sort(["instance", "__name__", "timestamp"])
    )

    # ── Compute rates for counter metrics, keep gauge as-is ───────────────────
    print("  Computing rates for counter metrics...")

    # Flag counters
    df = df.with_columns(
        pl.col("__name__").is_in(list(COUNTER_METRICS)).alias("is_counter")
    )

    # Vectorized diff for all groups at once
    df = df.with_columns([
        pl.col("value").diff().over(["instance", "__name__"]).alias("dv"),
        pl.col("timestamp").diff().dt.total_seconds()
          .over(["instance", "__name__"]).alias("dt_sec"),
    ])

    # For counters: use rate; for gauges: use value directly
    df = df.with_columns(
        pl.when(pl.col("is_counter"))
          .then((pl.col("dv") / pl.col("dt_sec")).clip(lower_bound=0.0))
          .otherwise(pl.col("value"))
          .alias("metric_value")
    ).filter(
        pl.when(pl.col("is_counter"))
          .then(pl.col("dt_sec") > 0)
          .otherwise(pl.lit(True))
    ).drop_nulls("metric_value").drop(["value", "dv", "dt_sec", "is_counter"])

    # ── Resample to 1-min for all (instance, hw_config, __name__) at once ─────
    print("  Resampling to 1-min...")
    df = df.sort(["instance", "hw_config", "__name__", "timestamp"])
    rs = (df
          .group_by_dynamic(
              "timestamp",
              every=RESAMPLE_S,
              closed="left",
              group_by=["instance", "hw_config", "__name__"],
          )
          .agg(pl.mean("metric_value"))
    )

    # ── Pivot metric names wide ───────────────────────────────────────────────
    print("  Pivoting metrics wide...")
    # Build readable column names: append _rate for counters, _now for gauge
    rs = rs.with_columns(
        pl.when(pl.col("__name__").is_in(list(COUNTER_METRICS)))
          .then(pl.concat_str([pl.col("__name__"), pl.lit("_rate")]))
          .otherwise(pl.concat_str([pl.col("__name__"), pl.lit("_now")]))
          .alias("col_name")
    )
    wide = rs.pivot(
        values="metric_value",
        index=["timestamp", "instance", "hw_config"],
        on="col_name",
        aggregate_function="first",
    )

    # Fill nulls (missing metrics for some nodes) with 0
    metric_cols = [c for c in wide.columns
                   if c not in ["timestamp", "instance", "hw_config"]]
    wide = wide.with_columns([pl.col(c).fill_null(0.0) for c in metric_cols])

    print(f"  Wide shape: {wide.shape}  |  metric cols: {len(metric_cols)}")

    # ── Split train / test ────────────────────────────────────────────────────
    ts     = wide["timestamp"]
    df_tr  = wide.filter((ts >= pl.lit(TRAIN_START)) & (ts <= pl.lit(TRAIN_END)))
    df_te  = wide.filter((ts >= pl.lit(TEST_START))  & (ts <= pl.lit(TEST_END)))

    n_train = len(df_tr)
    n_test  = len(df_te)

    df_tr.write_parquet(str(train_path), compression="snappy")
    df_te.write_parquet(str(test_path),  compression="snappy")

    tr_mb = round(train_path.stat().st_size / 1e6, 1)
    te_mb = round(test_path.stat().st_size  / 1e6, 1)

    print(f"\n  Disk splits done:")
    print(f"    Train rows: {n_train:,}  -> {train_path}  ({tr_mb} MB)")
    print(f"    Test  rows: {n_test:,}   -> {test_path}  ({te_mb} MB)")

    if train_path.exists():
        sample = pd.read_parquet(train_path, columns=["hw_config"])
        print(f"    hw_config dist: {sample['hw_config'].value_counts().to_dict()}")

    return {"train_rows": n_train, "test_rows": n_test, "metric_cols": metric_cols}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_disk_features(split_path: Path, out_path: Path, tag: str) -> Tuple[int, List[str]]:
    if not split_path.exists():
        print(f"  [SKIP] {split_path} not found")
        return 0, []

    df = pl.read_parquet(str(split_path))
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    print(f"\n  [{tag}] Input: {df.shape}  nodes: {df['instance'].n_unique()}")

    meta     = {"instance", "timestamp", "hw_config"}
    sig_cols = [c for c in df.columns if c not in meta]

    df = df.sort(["instance", "timestamp"])

    feat_exprs = []
    feat_names = []

    for sig in sig_cols:
        safe = sig.replace("/", "_").replace("-", "_")
        for win_min, ws in zip(ROLLING_WINS, ["5m", "15m", "60m"]):
            feat_exprs += [
                pl.col(sig).rolling_mean(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{ws}_mean"),
                pl.col(sig).rolling_std(window_size=win_min, min_periods=1)
                  .over("instance").fill_null(0.0).alias(f"{safe}_{ws}_std"),
                pl.col(sig).rolling_min(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{ws}_min"),
                pl.col(sig).rolling_max(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{ws}_max"),
            ]
            feat_names += [f"{safe}_{ws}_{a}" for a in ["mean","std","min","max"]]
        feat_exprs.append(
            pl.col(sig).diff(1).over("instance").alias(f"{safe}_lag1_diff")
        )
        feat_names.append(f"{safe}_lag1_diff")

    print(f"  [{tag}] Computing {len(feat_names)} features via Polars...")
    result = df.with_columns(feat_exprs)

    n_before  = len(result)
    nan_frac  = result.select(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Float32) for c in feat_names])
        .alias("nc")
    )["nc"] / max(len(feat_names), 1)
    result = result.filter(nan_frac <= NAN_THRESH)
    dropped = n_before - len(result)
    print(f"  [{tag}] Output: {result.shape}  | NaN-dropped: {dropped:,}")

    out_path.unlink(missing_ok=True)
    result.write_parquet(str(out_path), compression="snappy")
    return len(result), feat_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def fill_scale(df: pd.DataFrame, cols: List[str], sc=None):
    X = df[cols].copy()
    for c in cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    X = X.to_numpy(dtype=float)
    if sc is None:
        sc = StandardScaler(); X = sc.fit_transform(X)
    else:
        X = sc.transform(X)
    return X, sc


def train_disk_models(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                      out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    excl   = {"instance", "timestamp", "hw_config"}
    fcols  = [c for c in df_tr.columns
              if c not in excl and pd.api.types.is_numeric_dtype(df_tr[c])]
    te_set = {c for c in df_te.columns if c not in excl} if not df_te.empty else set(fcols)
    fcols  = [c for c in fcols if c in te_set]
    print(f"\n  Feature columns: {len(fcols)}")

    summary: Dict[str, Any] = {"per_farm": {}}
    all_scores: List[pd.DataFrame] = []

    for hw in sorted(df_tr["hw_config"].unique()):
        gtr = df_tr[df_tr["hw_config"] == hw]
        gte = df_te[df_te["hw_config"] == hw] if not df_te.empty else pd.DataFrame()
        n   = len(gtr)
        if n < MIN_ROWS:
            print(f"\n  [{hw}] SKIP — only {n} rows"); continue

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  train={n:,}  test={len(gte):,}")

        Xtr, sc = fill_scale(gtr, fcols)
        Xte, _  = fill_scale(gte, fcols, sc) if not gte.empty else (np.array([]), sc)

        pca    = PCA(n_components=PCA_VAR, svd_solver="full")
        Xp_tr  = pca.fit_transform(Xtr)
        Xp_te  = pca.transform(Xte) if Xte.shape[0] > 0 else np.array([])
        print(f"  [{hw}] PCA: {len(fcols)} -> {Xp_tr.shape[1]} components")

        print(f"  [{hw}] Training IsolationForest...")
        clf = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
        clf.fit(Xp_tr)
        if_tr_sc  = -clf.score_samples(Xp_tr)
        if_tr_fl  = clf.predict(Xp_tr) == -1
        if_tr_pct = if_tr_fl.mean()
        print(f"  [{hw}] IF train flagged: {if_tr_pct*100:.2f}%")
        if if_tr_pct > IF_LIMIT:
            print(f"  [{hw}] IF >15% — skipping farm"); continue

        if_te_sc = if_te_fl = None
        if Xp_te.shape[0] > 0:
            if_te_sc  = -clf.score_samples(Xp_te)
            if_te_fl  = clf.predict(Xp_te) == -1
            print(f"  [{hw}] IF test  flagged: {if_te_fl.mean()*100:.2f}%")

        print(f"  [{hw}] Training Autoencoder...")
        ae = ReconAE(Xtr.shape[1]); ae.fit(Xtr)
        ae_tr_e, ae_tr_fl = ae.score(Xtr)
        print(f"  [{hw}] AE train flagged: {ae_tr_fl.mean()*100:.2f}%  thr={ae.thr:.4f}")

        ae_te_e = ae_te_fl = None
        if Xte.shape[0] > 0:
            ae_te_e, ae_te_fl = ae.score(Xte)
            print(f"  [{hw}] AE test  flagged: {ae_te_fl.mean()*100:.2f}%")

        joblib.dump(sc,  out_dir / f"disk_scaler_{hw}.joblib")
        joblib.dump(pca, out_dir / f"disk_pca_{hw}.joblib")
        joblib.dump(clf, out_dir / f"disk_isoforest_{hw}.joblib")
        joblib.dump(ae,  out_dir / f"disk_autoencoder_{hw}.joblib")
        print(f"  [{hw}] Saved: scaler, pca, isoforest, autoencoder")

        meta_cols = ["instance", "timestamp", "hw_config"]
        tr_out = gtr[meta_cols].copy().reset_index(drop=True)
        tr_out["if_score"] = if_tr_sc; tr_out["if_flag"] = if_tr_fl
        tr_out["ae_score"] = ae_tr_e;  tr_out["ae_flag"] = ae_tr_fl
        tr_out["split"] = "train";  all_scores.append(tr_out)

        if if_te_fl is not None:
            te_out = gte[meta_cols].copy().reset_index(drop=True)
            te_out["if_score"] = if_te_sc; te_out["if_flag"] = if_te_fl
            te_out["ae_score"] = ae_te_e;  te_out["ae_flag"] = ae_te_fl
            te_out["split"] = "test";  all_scores.append(te_out)

        summary["per_farm"][hw] = {
            "train_rows":        int(n),
            "test_rows":         int(len(gte)),
            "pca_components":    int(Xp_tr.shape[1]),
            "if_train_flag_pct": round(float(if_tr_pct)*100, 3),
            "if_test_flag_pct":  round(float(if_te_fl.mean())*100, 3) if if_te_fl is not None else None,
            "ae_thr":            round(ae.thr, 6),
            "ae_train_flag_pct": round(float(ae_tr_fl.mean())*100, 3),
            "ae_test_flag_pct":  round(float(ae_te_fl.mean())*100, 3) if ae_te_fl is not None else None,
        }

    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
        sp = out_dir / "disk_anomaly_scores.parquet"
        scores_df.to_parquet(sp, index=False, compression="snappy")
        print(f"\n  Anomaly scores -> {sp}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir",     type=Path, default=Path("structural_clean"))
    ap.add_argument("--splits-dir",    type=Path, default=Path("artifacts/splits"))
    ap.add_argument("--features-dir",  type=Path, default=Path("artifacts/features"))
    ap.add_argument("--models-dir",    type=Path, default=Path("artifacts/models"))
    ap.add_argument("--skip-splits",   action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()

    for d in [args.splits_dir, args.features_dir, args.models_dir]:
        ensure_dir(d)

    summary: Dict[str, Any] = {
        "pipeline":     "disk_data_polars",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.skip_splits:
        summary["splits"] = build_disk_splits(args.clean_dir, args.splits_dir)
    else:
        print("  [SKIP] Splits")

    tr_feat = args.features_dir / "disk_data_train_features.parquet"
    te_feat = args.features_dir / "disk_data_test_features.parquet"

    if not args.skip_features:
        print(f"\n{'='*60}\n  STEP 2 — DISK FEATURES\n{'='*60}")
        tr_rows, fnames = build_disk_features(
            args.splits_dir / "disk_data_train.parquet", tr_feat, "TRAIN")
        te_rows, _ = build_disk_features(
            args.splits_dir / "disk_data_test.parquet",  te_feat, "TEST")
        summary["features"] = {"train_rows": tr_rows, "test_rows": te_rows,
                                "n_features": len(fnames)}
    else:
        print("  [SKIP] Features")

    print(f"\n{'='*60}\n  STEP 3 — DISK TRAINING\n{'='*60}")
    if not tr_feat.exists():
        print(f"  [ERROR] {tr_feat} missing"); return

    df_tr = pd.read_parquet(tr_feat)
    df_te = pd.read_parquet(te_feat) if te_feat.exists() else pd.DataFrame()
    print(f"  Train: {df_tr.shape}  |  Test: {df_te.shape}")

    summary["models"] = train_disk_models(df_tr, df_te, args.models_dir)

    sp = args.models_dir / "disk_training_summary.json"
    write_json(sp, summary)

    print(f"\n{'='*60}")
    print("  DISK PIPELINE COMPLETE")
    print(f"  Summary -> {sp}")
    print(f"{'='*60}")
    for hw, info in summary["models"].get("per_farm", {}).items():
        ift = info.get("if_test_flag_pct")
        aet = info.get("ae_test_flag_pct")
        line = f"    {hw}: train={info['train_rows']:,}"
        line += f"  IF_test={ift:.1f}%  AE_test={aet:.1f}%" if ift is not None else "  (no test data)"
        print(line)


if __name__ == "__main__":
    main()
