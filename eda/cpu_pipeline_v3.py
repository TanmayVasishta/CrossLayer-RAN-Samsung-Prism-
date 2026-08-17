"""
eda/cpu_pipeline_v3.py  — Polars-accelerated CPU pipeline
==========================================================
Uses Polars for all heavy data manipulation (sort, groupby, diff, resample).
Polars is typically 10-50x faster than pandas for this class of operations
because it uses Rust internals with multi-threaded execution.

Pipeline:
  Step 1 — Splits  : Stream 115 cpu parquets, sum cores, diff for rate,
                     group_by_dynamic resample to 1-min, pivot modes wide,
                     write train / test splits incrementally.
  Step 2 — Features: Rolling window (5/15/60min) + lag-diff per node.
  Step 3 — Train   : IsolationForest + Autoencoder per hw_config (sklearn).
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
import polars as pl
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START = datetime(2023, 5, 19,  0,  0,  0)
TRAIN_END   = datetime(2023, 5, 22, 23, 59, 59)
TEST_START  = datetime(2023, 5, 23,  0,  0,  0)
TEST_END    = datetime(2023, 5, 23, 23, 59, 59)

HW_CONFIGS  = {"farm14", "farm16", "farm18", "farm19", "farm23"}
KEEP_MODES  = ["user", "system", "idle", "iowait", "irq", "softirq"]
RESAMPLE_S  = "1m"   # Polars duration string: 1 minute

ROLLING_WINS_PD = ["5min", "15min", "60min"]   # pandas rolling strings
NAN_THRESH      = 0.30
IF_CONTAM       = 0.05
IF_LIMIT        = 0.15
PCA_VAR         = 0.95
MIN_ROWS        = 200
AE_SIGMA        = 3.0


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def hw_config_from_instance(inst: str) -> str:
    c = str(inst).split(":")[0][:6].lower()
    return c if c in HW_CONFIGS else "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FAST POLARS CPU SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def process_file_polars(f: Path) -> Optional[pl.DataFrame]:
    """
    Load one cpu parquet with Polars, sum CPU cores, compute rate via diff,
    resample to 1-min, pivot modes wide, add utilisation ratios.

    Returns a Polars DataFrame with columns:
      timestamp, instance, hw_config,
      cpu_idle_rate, cpu_user_rate, cpu_system_rate,
      cpu_iowait_rate, cpu_irq_rate, cpu_softirq_rate,
      cpu_total_rate,
      cpu_idle_util, cpu_user_util, ...
    Only rows in [TRAIN_START, TEST_END] retained.
    """
    try:
        lf = pl.scan_parquet(str(f))
    except Exception as e:
        print(f"    [WARN] {f.name}: {e}")
        return None

    # Filter time range + modes early (lazy, pushed down to scan)
    lf = (lf
          .filter(pl.col("mode").is_in(KEEP_MODES))
          .filter(pl.col("timestamp") >= pl.lit(TRAIN_START))
          .filter(pl.col("timestamp") <= pl.lit(TEST_END))
    )

    # Collect only needed columns
    try:
        df = lf.select(["instance", "mode", "timestamp", "value"]).collect()
    except Exception as e:
        print(f"    [WARN] collect {f.name}: {e}")
        return None

    if df.is_empty():
        return None

    # Add hw_config, filter unknowns
    df = df.with_columns(
        pl.col("instance").map_elements(hw_config_from_instance, return_dtype=pl.String)
          .alias("hw_config")
    ).filter(pl.col("hw_config") != "unknown")

    if df.is_empty():
        return None

    # ── Sum across CPU cores: group by (instance, hw_config, mode, timestamp) ──
    # Each file represents one cpu core number; we sum over all cores within
    # the same (instance, mode, timestamp) — this happens implicitly since
    # each file has only one cpu column value but we want total across all files.
    # Within a single file: one (instance, mode, timestamp) → one row already.
    # We aggregate anyway to handle any duplicates.
    df = (df
          .group_by(["instance", "hw_config", "mode", "timestamp"])
          .agg(pl.sum("value"))
          .sort(["instance", "mode", "timestamp"])
    )

    # ── Vectorized diff for rate ───────────────────────────────────────────────
    # diff(value) / diff(timestamp_seconds) per (instance, mode) group
    df = df.with_columns([
        pl.col("value").diff().over(["instance", "mode"]).alias("dv"),
        (pl.col("timestamp").diff().dt.total_seconds()
           .over(["instance", "mode"])).alias("dt_sec"),
    ])

    df = (df
          .with_columns(
              (pl.col("dv") / pl.col("dt_sec")).clip(lower_bound=0.0).alias("cpu_rate")
          )
          .drop_nulls("cpu_rate")
          .filter(pl.col("dt_sec") > 0)
          .drop(["value", "dv", "dt_sec"])
    )

    if df.is_empty():
        return None

    # ── Resample to 1-min via group_by_dynamic (single Polars call, no Python loop) ─
    # Sort first — required by group_by_dynamic
    df = df.sort(["instance", "hw_config", "mode", "timestamp"])

    rs_all = (
        df
        .group_by_dynamic(
            "timestamp",
            every=RESAMPLE_S,
            closed="left",
            group_by=["instance", "hw_config", "mode"],
        )
        .agg(pl.mean("cpu_rate"))
    )

    if rs_all.is_empty():
        return None

    # ── Pivot modes → wide ────────────────────────────────────────────────────
    wide = rs_all.pivot(
        values="cpu_rate",
        index=["timestamp", "instance", "hw_config"],
        on="mode",
        aggregate_function="first",
    )

    # Rename mode columns to cpu_*_rate
    rename_map = {m: f"cpu_{m}_rate" for m in KEEP_MODES if m in wide.columns}
    if rename_map:
        wide = wide.rename(rename_map)

    # Fill nulls in rate columns with 0 (mode not present = 0 usage)
    rate_cols = [f"cpu_{m}_rate" for m in KEEP_MODES if f"cpu_{m}_rate" in wide.columns]
    wide = wide.with_columns([pl.col(c).fill_null(0.0) for c in rate_cols])

    # ── Total rate + utilisation ratios ───────────────────────────────────────
    total_expr = sum(pl.col(c) for c in rate_cols)
    wide = wide.with_columns(total_expr.alias("cpu_total_rate"))

    util_exprs = [
        pl.when(pl.col("cpu_total_rate") > 0)
          .then(pl.col(c) / pl.col("cpu_total_rate"))
          .otherwise(pl.lit(None))
          .alias(c.replace("_rate", "_util"))
        for c in rate_cols
    ]
    wide = wide.with_columns(util_exprs)

    return wide


def build_cpu_splits(clean_dir: Path, out_dir: Path) -> Dict[str, Any]:
    cpu_dir = clean_dir / "cpu_data"
    files   = sorted(cpu_dir.glob("*.parquet"))
    if not files:
        print(f"[ERROR] No parquets in {cpu_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  STEP 1 — CPU SPLITS (Polars)  |  {len(files)} files")
    print(f"{'='*60}")

    ensure_dir(out_dir)
    train_path = out_dir / "cpu_data_train.parquet"
    test_path  = out_dir / "cpu_data_test.parquet"
    train_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    tr_frames: List[pl.DataFrame] = []
    te_frames: List[pl.DataFrame] = []
    n_train = n_test = 0
    flush_every = 10   # write to disk every N files to control RAM

    for i, f in enumerate(files):
        pct = 100 * (i + 1) / len(files)
        print(f"  [{i+1:3d}/{len(files)}  {pct:5.1f}%]  {f.name}", flush=True)

        wide = process_file_polars(f)
        if wide is None or wide.is_empty():
            print(f"    → empty/skipped")
            continue

        ts = wide["timestamp"]
        df_tr = wide.filter((ts >= pl.lit(TRAIN_START)) & (ts <= pl.lit(TRAIN_END)))
        df_te = wide.filter((ts >= pl.lit(TEST_START))  & (ts <= pl.lit(TEST_END)))

        if not df_tr.is_empty():
            tr_frames.append(df_tr)
            n_train += len(df_tr)
        if not df_te.is_empty():
            te_frames.append(df_te)
            n_test += len(df_te)

        # Flush to disk every flush_every files
        if (i + 1) % flush_every == 0:
            if tr_frames:
                combined_tr = pl.concat(tr_frames, rechunk=False)
                _append_parquet(combined_tr.to_pandas(), train_path)
                tr_frames = []
                print(f"    → Flushed train ({n_train:,} rows so far)")
            if te_frames:
                combined_te = pl.concat(te_frames, rechunk=False)
                _append_parquet(combined_te.to_pandas(), test_path)
                te_frames = []
            gc.collect()

    # Final flush
    if tr_frames:
        _append_parquet(pl.concat(tr_frames, rechunk=False).to_pandas(), train_path)
    if te_frames:
        _append_parquet(pl.concat(te_frames, rechunk=False).to_pandas(), test_path)

    print(f"\n  ✓ CPU splits done:")
    print(f"    Train rows: {n_train:,}  → {train_path}  "
          f"({train_path.stat().st_size / 1e6:.1f} MB)" if train_path.exists() else "")
    print(f"    Test  rows: {n_test:,}   → {test_path}  "
          f"({test_path.stat().st_size / 1e6:.1f} MB)" if test_path.exists() else "")

    if train_path.exists():
        sample = pd.read_parquet(train_path, columns=["hw_config"])
        hw_dist = sample["hw_config"].value_counts().to_dict()
        print(f"  Train hw_config dist: {hw_dist}")

    return {"train_rows": n_train, "test_rows": n_test}


def _append_parquet(df_pd: pd.DataFrame, path: Path) -> None:
    """Append a pandas DataFrame to a parquet file (schema-compatible)."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df_pd], ignore_index=True)
        combined.to_parquet(path, index=False, compression="snappy")
    else:
        df_pd.to_parquet(path, index=False, compression="snappy")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING (Polars rolling)
# ═══════════════════════════════════════════════════════════════════════════════

def build_cpu_features(split_path: Path, out_path: Path, tag: str) -> Tuple[int, List[str]]:
    """Rolling window + lag-diff on CPU data using Polars over() expressions."""
    if not split_path.exists():
        print(f"  [SKIP] {split_path} not found")
        return 0, []

    df = pl.read_parquet(str(split_path))
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    print(f"\n  [{tag}] Input: {df.shape}  nodes: {df['instance'].n_unique()}")

    meta = {"instance", "timestamp", "hw_config"}
    sig_cols = [c for c in df.columns if c not in meta]

    # Sort for rolling correctness
    df = df.sort(["instance", "timestamp"])

    feat_exprs = []
    feat_names = []

    for sig in sig_cols:
        safe = sig.replace("/", "_").replace("-", "_")

        # Rolling windows — using group_by_dynamic per instance is expensive;
        # instead use polars rolling_mean/std over sorted data grouped by instance
        for win_min, win_str in [(5, "5m"), (15, "15m"), (60, "60m")]:
            win_pd = f"{win_min}min"
            # Polars rolling with group_by: use rolling().over() — available in newer polars
            feat_exprs += [
                pl.col(sig).rolling_mean(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{win_str}_mean"),
                pl.col(sig).rolling_std(window_size=win_min, min_periods=1)
                  .over("instance").fill_null(0.0).alias(f"{safe}_{win_str}_std"),
                pl.col(sig).rolling_min(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{win_str}_min"),
                pl.col(sig).rolling_max(window_size=win_min, min_periods=1)
                  .over("instance").alias(f"{safe}_{win_str}_max"),
            ]
            feat_names += [f"{safe}_{win_str}_{a}" for a in ["mean", "std", "min", "max"]]

        # Lag-1 diff
        feat_exprs.append(
            pl.col(sig).diff(1).over("instance").alias(f"{safe}_lag1_diff")
        )
        feat_names.append(f"{safe}_lag1_diff")

    print(f"  [{tag}] Computing {len(feat_names)} features via Polars...")
    result = df.with_columns(feat_exprs)

    # Drop rows with >30% NaN features
    n_before = len(result)
    nan_frac = result.select(
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.Float32) for c in feat_names])
        .alias("null_count")
    )["null_count"] / len(feat_names)
    result = result.filter(nan_frac <= NAN_THRESH)
    dropped = n_before - len(result)
    print(f"  [{tag}] Output: {result.shape}  |  NaN-dropped: {dropped:,}")

    # Write to parquet
    out_path.unlink(missing_ok=True)
    result.write_parquet(str(out_path), compression="snappy")
    return len(result), feat_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING (sklearn — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class ReconAE:
    def __init__(self, n: int):
        h1 = min(64, max(n, 4)); h2 = min(16, max(n // 4, 2))
        self.sc = StandardScaler()
        self.m  = MLPRegressor(
            hidden_layer_sizes=(h1, h2, h1), activation="relu", solver="adam",
            learning_rate_init=1e-3, max_iter=30, batch_size=256,
            random_state=42, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, verbose=False,
        )
        self.thr = self.mu = self.sd = 0.0

    def fit(self, X):
        Xs = self.sc.fit_transform(X); self.m.fit(Xs, Xs)
        e  = self._err(Xs)
        self.mu, self.sd = float(e.mean()), float(e.std())
        self.thr = self.mu + AE_SIGMA * self.sd
        return self

    def _err(self, Xs): return np.mean((Xs - self.m.predict(Xs)) ** 2, axis=1)

    def score(self, X):
        Xs = self.sc.transform(X); e = self._err(Xs)
        return e, e > self.thr


def fill_scale(df, cols, sc=None):
    X = df[cols].copy()
    for c in cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    X = X.to_numpy(dtype=float)
    if sc is None: sc = StandardScaler(); X = sc.fit_transform(X)
    else:          X = sc.transform(X)
    return X, sc


def train_cpu_models(df_tr: pd.DataFrame, df_te: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    excl  = {"instance", "timestamp", "hw_config"}
    fcols_tr = [c for c in df_tr.columns if c not in excl and pd.api.types.is_numeric_dtype(df_tr[c])]
    fcols_te = {c for c in df_te.columns  if c not in excl and pd.api.types.is_numeric_dtype(df_te[c])} if not df_te.empty else set(fcols_tr)
    fcols = [c for c in fcols_tr if c in fcols_te]
    print(f"\n  Feature columns: {len(fcols)}")

    summary: Dict[str, Any] = {"per_farm": {}}
    all_scores: List[pd.DataFrame] = []

    for hw in sorted(df_tr["hw_config"].unique()):
        gtr = df_tr[df_tr["hw_config"] == hw]
        gte = df_te[df_te["hw_config"] == hw] if not df_te.empty else pd.DataFrame()
        n   = len(gtr)

        if n < MIN_ROWS:
            print(f"\n  [{hw}] SKIP — {n} rows < {MIN_ROWS}"); continue

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  train={n:,}  test={len(gte):,}")

        Xtr, sc = fill_scale(gtr, fcols)
        Xte, _  = fill_scale(gte, fcols, sc) if not gte.empty else (np.array([]), sc)

        pca   = PCA(n_components=PCA_VAR, svd_solver="full")
        Xp_tr = pca.fit_transform(Xtr)
        Xp_te = pca.transform(Xte) if Xte.shape[0] > 0 else np.array([])
        print(f"  [{hw}] PCA: {len(fcols)} → {Xp_tr.shape[1]} components")

        print(f"  [{hw}] Training IsolationForest...")
        clf = IsolationForest(contamination=IF_CONTAM, random_state=42, n_jobs=-1)
        clf.fit(Xp_tr)
        if_tr_sc = -clf.score_samples(Xp_tr); if_tr_fl = clf.predict(Xp_tr) == -1
        if_tr_pct = if_tr_fl.mean()
        print(f"  [{hw}] IF train flagged: {if_tr_pct*100:.2f}%")
        if if_tr_pct > IF_LIMIT:
            print(f"  [{hw}] ⚠ IF > 15% — skipping"); continue

        if_te_sc = if_te_fl = None
        if Xp_te.shape[0] > 0:
            if_te_sc = -clf.score_samples(Xp_te); if_te_fl = clf.predict(Xp_te) == -1
            print(f"  [{hw}] IF test  flagged: {if_te_fl.mean()*100:.2f}%")

        print(f"  [{hw}] Training Autoencoder...")
        ae = ReconAE(Xtr.shape[1]); ae.fit(Xtr)
        ae_tr_e, ae_tr_fl = ae.score(Xtr)
        print(f"  [{hw}] AE train flagged: {ae_tr_fl.mean()*100:.2f}%  thr={ae.thr:.4f}")

        ae_te_e = ae_te_fl = None
        if Xte.shape[0] > 0:
            ae_te_e, ae_te_fl = ae.score(Xte)
            print(f"  [{hw}] AE test  flagged: {ae_te_fl.mean()*100:.2f}%")

        joblib.dump(sc,  out_dir / f"cpu_scaler_{hw}.joblib")
        joblib.dump(pca, out_dir / f"cpu_pca_{hw}.joblib")
        joblib.dump(clf, out_dir / f"cpu_isoforest_{hw}.joblib")
        joblib.dump(ae,  out_dir / f"cpu_autoencoder_{hw}.joblib")
        print(f"  [{hw}] ✓ Saved: scaler, pca, isoforest, autoencoder")

        meta_cols = ["instance", "timestamp", "hw_config"]
        tr_out = gtr[meta_cols].copy().reset_index(drop=True)
        tr_out["if_score"] = if_tr_sc; tr_out["if_flag"] = if_tr_fl
        tr_out["ae_score"] = ae_tr_e;  tr_out["ae_flag"] = ae_tr_fl
        tr_out["split"] = "train"; all_scores.append(tr_out)

        if if_te_fl is not None:
            te_out = gte[meta_cols].copy().reset_index(drop=True)
            te_out["if_score"] = if_te_sc; te_out["if_flag"] = if_te_fl
            te_out["ae_score"] = ae_te_e;  te_out["ae_flag"] = ae_te_fl
            te_out["split"] = "test"; all_scores.append(te_out)

        summary["per_farm"][hw] = {
            "train_rows":        int(n), "test_rows": int(len(gte)),
            "pca_components":    int(Xp_tr.shape[1]),
            "if_train_flag_pct": round(float(if_tr_pct)*100, 3),
            "if_test_flag_pct":  round(float(if_te_fl.mean())*100, 3) if if_te_fl is not None else None,
            "ae_thr":            round(ae.thr, 6),
            "ae_train_flag_pct": round(float(ae_tr_fl.mean())*100, 3),
            "ae_test_flag_pct":  round(float(ae_te_fl.mean())*100, 3) if ae_te_fl is not None else None,
        }

    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
        sp = out_dir / "cpu_anomaly_scores.parquet"
        scores_df.to_parquet(sp, index=False, compression="snappy")
        print(f"\n  ✓ Anomaly scores → {sp}")

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
        "pipeline": "cpu_data_v3_polars",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # Step 1
    if not args.skip_splits:
        summary["splits"] = build_cpu_splits(args.clean_dir, args.splits_dir)
    else:
        print("  [SKIP] Splits — using existing")

    # Step 2
    tr_feat = args.features_dir / "cpu_data_train_features.parquet"
    te_feat = args.features_dir / "cpu_data_test_features.parquet"
    if not args.skip_features:
        print(f"\n{'='*60}\n  STEP 2 — CPU FEATURES (Polars)\n{'='*60}")
        tr_rows, fnames = build_cpu_features(args.splits_dir / "cpu_data_train.parquet", tr_feat, "TRAIN")
        te_rows, _      = build_cpu_features(args.splits_dir / "cpu_data_test.parquet",  te_feat, "TEST")
        summary["features"] = {"train_rows": tr_rows, "test_rows": te_rows, "n_features": len(fnames)}
    else:
        print("  [SKIP] Features — using existing")

    # Step 3
    print(f"\n{'='*60}\n  STEP 3 — CPU TRAINING\n{'='*60}")
    if not tr_feat.exists():
        print(f"  [ERROR] {tr_feat} missing"); return

    df_tr = pd.read_parquet(tr_feat)
    df_te = pd.read_parquet(te_feat) if te_feat.exists() else pd.DataFrame()
    print(f"  Train: {df_tr.shape}  |  Test: {df_te.shape}")

    summary["models"] = train_cpu_models(df_tr, df_te, args.models_dir)

    sp = args.models_dir / "cpu_training_summary.json"
    write_json(sp, summary)

    print(f"\n{'='*60}")
    print("  ✓  CPU PIPELINE v3 (Polars) COMPLETE")
    print(f"  Summary → {sp}")
    print(f"{'='*60}")
    for hw, info in summary["models"].get("per_farm", {}).items():
        ift = info.get("if_test_flag_pct")
        aet = info.get("ae_test_flag_pct")
        s   = f"    {hw}: train={info['train_rows']:,}"
        s  += f"  IF_test={ift:.1f}%  AE_test={aet:.1f}%" if ift is not None else "  (no test rows)"
        print(s)


if __name__ == "__main__":
    main()
