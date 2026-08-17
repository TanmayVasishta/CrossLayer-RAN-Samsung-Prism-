"""
eda/cpu_pipeline_v2.py
======================
FAST vectorized CPU anomaly detection pipeline.

Key optimizations vs v1:
  - File-by-file streaming: never loads all 115 files at once
  - Vectorized diff() via sort + groupby — no Python loops per group
  - Aggregate across CPU cores (sum) BEFORE groupby operations
  - Write splits incrementally to parquet using append mode
  - Feature engineering operates on the already-aggregated per-node data

Schema (structural_clean/cpu_data/*.parquet):
  [__name__, cpu, instance, job, mode, timestamp, value]
  - cpu      : core number (int)
  - mode     : idle/user/system/iowait/irq/softirq/steal/guest...
  - value    : cumulative CPU seconds (counter → diff for rate)

Strategy:
  1. Per file: sum across all CPU cores (cpu col) → per-node per-mode total
  2. Sort by (instance, mode, timestamp) → vectorized diff for rate
  3. Resample per (instance, mode) to 1-min → pivot modes wide
  4. Compute utilisation ratios
  5. Filter to train/test time windows
  6. Append to output parquets
  7. Feature engineering on concatenated splits
  8. Train IF + AE per hw_config
"""
from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─── Paths & Config ────────────────────────────────────────────────────────────
TRAIN_START = pd.Timestamp("2023-05-19 00:00:00", tz=None)
TRAIN_END   = pd.Timestamp("2023-05-22 23:59:59.999999999", tz=None)
TEST_START  = pd.Timestamp("2023-05-23 00:00:00", tz=None)
TEST_END    = pd.Timestamp("2023-05-23 23:59:59.999999999", tz=None)

HW_CONFIGS  = {"farm14", "farm16", "farm18", "farm19", "farm23"}
KEEP_MODES  = {"user", "system", "idle", "iowait", "irq", "softirq"}
RESAMPLE    = "1min"

ROLLING_WINS = ["5min", "15min", "60min"]
ROLLING_AGGS = ["mean", "std", "min", "max"]
NAN_THRESH   = 0.30

IF_CONTAMINATION = 0.05
IF_FLAG_LIMIT    = 0.15
PCA_VAR          = 0.95
MIN_ROWS         = 200
AE_SIGMA         = 3.0


def hw_config(inst: str) -> str:
    c = str(inst).split(":")[0][:6].lower()
    return c if c in HW_CONFIGS else "unknown"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FAST VECTORIZED CPU SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def process_one_file(f: Path) -> Optional[pd.DataFrame]:
    """
    Load one cpu parquet, sum across cores, compute rate, resample to 1-min.
    Returns a DataFrame with columns:
      [timestamp, instance, hw_config, cpu_user_rate, cpu_system_rate,
       cpu_idle_rate, cpu_iowait_rate, cpu_irq_rate, cpu_softirq_rate,
       cpu_total_rate, cpu_user_util, cpu_system_util, ...]
    Only rows in [TRAIN_START, TEST_END] are kept.
    """
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        print(f"    [WARN] {f.name}: {e}")
        return None

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Filter to time window before doing heavy work
    mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TEST_END)
    df = df[mask]
    if df.empty:
        return None

    # Add hw_config, filter unknown
    df["hw_config"] = df["instance"].apply(hw_config)
    df = df[df["hw_config"] != "unknown"]
    if df.empty:
        return None

    # Filter to desired modes
    if "mode" in df.columns:
        df = df[df["mode"].isin(KEEP_MODES)]
    if df.empty:
        return None

    # ── Sum across CPU cores (vectorized) ──────────────────────────────────────
    # Group by (instance, mode, timestamp) and sum value across all cpu cores
    # This reduces rows from N_cores × N_modes × N_timestamps
    #                       to         N_modes × N_timestamps per node
    agg = (df.groupby(["instance", "hw_config", "mode", "timestamp"], sort=False)["value"]
             .sum()
             .reset_index())

    # ── Sort + vectorized diff for rate ────────────────────────────────────────
    agg = agg.sort_values(["instance", "mode", "timestamp"])
    grp = agg.groupby(["instance", "mode"], sort=False)

    dt_sec = grp["timestamp"].diff().dt.total_seconds()
    dv     = grp["value"].diff()
    rate   = (dv / dt_sec).clip(lower=0.0)   # clip negatives (counter resets)

    agg["cpu_rate"] = rate.values
    agg = agg.dropna(subset=["cpu_rate"])

    # ── Resample per (instance, mode) to 1-min ─────────────────────────────────
    # Use pandas resample on a MultiIndex — much faster than Python loops
    agg = agg.set_index("timestamp")
    resampled_parts = []

    for (inst, mode), grp_df in agg.groupby(["instance", "mode"], sort=False):
        hw = grp_df["hw_config"].iloc[0]
        rs = (grp_df[["cpu_rate"]]
                .resample(RESAMPLE)
                .mean()
                .ffill(limit=3)
                .dropna())
        rs["instance"]  = inst
        rs["hw_config"] = hw
        rs["mode"]      = mode
        resampled_parts.append(rs)

    if not resampled_parts:
        return None

    rs_all = pd.concat(resampled_parts).reset_index()

    # ── Pivot modes → wide per-node per-timestamp ──────────────────────────────
    wide = rs_all.pivot_table(
        index=["timestamp", "instance", "hw_config"],
        columns="mode",
        values="cpu_rate",
        aggfunc="first"
    ).reset_index()

    # Rename columns: idle → cpu_idle_rate, etc.
    rename = {m: f"cpu_{m}_rate" for m in KEEP_MODES if m in wide.columns}
    wide = wide.rename(columns=rename)
    wide.columns.name = None

    # ── Utilisation ratios ────────────────────────────────────────────────────
    rate_cols = [c for c in wide.columns if c.endswith("_rate")]
    wide["cpu_total_rate"] = wide[rate_cols].sum(axis=1)
    for col in rate_cols:
        util_col = col.replace("_rate", "_util")
        wide[util_col] = np.where(
            wide["cpu_total_rate"] > 0,
            wide[col] / wide["cpu_total_rate"],
            np.nan
        )

    return wide


def build_cpu_splits(clean_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Stream through all CPU parquets file-by-file, build and write splits."""
    cpu_dir = clean_dir / "cpu_data"
    files   = sorted(cpu_dir.glob("*.parquet"))
    if not files:
        print(f"[ERROR] No parquets in {cpu_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  STEP 1 — CPU SPLITS  ({len(files)} files)")
    print(f"{'='*60}")

    ensure_dir(out_dir)
    train_path = out_dir / "cpu_data_train.parquet"
    test_path  = out_dir / "cpu_data_test.parquet"

    # Remove stale output files
    train_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)

    writer_tr: Optional[pq.ParquetWriter] = None
    writer_te: Optional[pq.ParquetWriter] = None
    n_train = n_test = 0
    schema = None

    for i, f in enumerate(files):
        pct = 100 * (i + 1) / len(files)
        print(f"  [{i+1:3d}/{len(files)}  {pct:5.1f}%]  {f.name}", flush=True)

        wide = process_one_file(f)
        if wide is None or wide.empty:
            print(f"    → empty/skipped")
            continue

        # Split into train / test
        ts = wide["timestamp"]
        df_tr = wide[(ts >= TRAIN_START) & (ts <= TRAIN_END)]
        df_te = wide[(ts >= TEST_START)  & (ts <= TEST_END)]

        # Write train
        if not df_tr.empty:
            tbl = pa.Table.from_pandas(df_tr, preserve_index=False)
            if writer_tr is None:
                schema = tbl.schema
                writer_tr = pq.ParquetWriter(str(train_path), schema, compression="snappy")
            writer_tr.write_table(tbl)
            n_train += len(df_tr)

        # Write test
        if not df_te.empty:
            tbl = pa.Table.from_pandas(df_te, preserve_index=False)
            if writer_te is None:
                if schema is None:
                    schema = tbl.schema
                writer_te = pq.ParquetWriter(str(test_path), schema, compression="snappy")
            writer_te.write_table(tbl)
            n_test += len(df_te)

        del wide, df_tr, df_te
        gc.collect()

    if writer_tr: writer_tr.close()
    if writer_te: writer_te.close()

    print(f"\n  ✓ CPU splits done:")
    print(f"    Train rows: {n_train:,}  → {train_path}")
    print(f"    Test  rows: {n_test:,}   → {test_path}")

    # Quick hw_config check
    if train_path.exists():
        sample = pd.read_parquet(train_path, columns=["hw_config"])
        print(f"  hw_config distribution (train):")
        print(f"  {sample['hw_config'].value_counts().to_dict()}")

    return {"train_rows": n_train, "test_rows": n_test}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_cpu_features(split_path: Path, out_path: Path, tag: str) -> Tuple[int, List[str]]:
    """Rolling window + lag-diff features on per-node CPU data."""
    if not split_path.exists():
        print(f"  [SKIP] {split_path} not found")
        return 0, []

    df = pd.read_parquet(split_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"\n  [{tag}] Input: {df.shape}  nodes: {df['instance'].nunique()}")

    meta_cols   = {"instance", "timestamp", "hw_config"}
    signal_cols = [c for c in df.columns if c not in meta_cols]

    out_path.unlink(missing_ok=True)
    writer = None
    total_rows    = 0
    total_dropped = 0
    feature_names: List[str] = []

    for inst, inst_df in df.groupby("instance", sort=False):
        hw = inst_df["hw_config"].iloc[0]
        inst_df = inst_df.set_index("timestamp").sort_index()
        inst_df = inst_df[~inst_df.index.duplicated(keep="first")]

        feat_cols: Dict[str, pd.Series] = {}
        for sig in signal_cols:
            if sig not in inst_df.columns:
                continue
            s    = inst_df[sig]
            safe = sig.replace("/", "_").replace("-", "_")

            for win in ROLLING_WINS:
                roller = s.rolling(window=win, min_periods=1)
                ws = win.replace("min", "m")
                feat_cols[f"{safe}_{ws}_mean"] = roller.mean()
                feat_cols[f"{safe}_{ws}_std"]  = roller.std().fillna(0.0)
                feat_cols[f"{safe}_{ws}_min"]  = roller.min()
                feat_cols[f"{safe}_{ws}_max"]  = roller.max()

            feat_cols[f"{safe}_lag1_diff"] = s.diff(1)

        if not feat_cols:
            continue

        wide = pd.DataFrame(feat_cols, index=inst_df.index).reset_index()
        wide["instance"]  = inst
        wide["hw_config"] = hw

        fn = [c for c in wide.columns if c not in ["instance", "timestamp", "hw_config"]]
        n_before = len(wide)
        nan_frac = wide[fn].isna().mean(axis=1)
        wide     = wide[nan_frac <= NAN_THRESH].reset_index(drop=True)
        total_dropped += n_before - len(wide)
        total_rows    += len(wide)

        if wide.empty:
            continue
        if not feature_names:
            feature_names = fn

        tbl = pa.Table.from_pandas(wide, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), tbl.schema, compression="snappy")
        writer.write_table(tbl)

    if writer:
        writer.close()

    print(f"  [{tag}] Output: {total_rows:,} rows × {len(feature_names)+3} cols  "
          f"| NaN-dropped: {total_dropped:,}")
    return total_rows, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class ReconAE:
    def __init__(self, n: int):
        h1 = min(64, max(n, 4))
        h2 = min(16, max(n // 4, 2))
        self.sc = StandardScaler()
        self.m  = MLPRegressor(
            hidden_layer_sizes=(h1, h2, h1), activation="relu",
            solver="adam", learning_rate_init=1e-3, max_iter=30,
            batch_size=256, random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5, verbose=False,
        )
        self.thr = self.mu = self.sd = 0.0

    def fit(self, X):
        Xs = self.sc.fit_transform(X)
        self.m.fit(Xs, Xs)
        e = self._err(Xs)
        self.mu, self.sd = float(e.mean()), float(e.std())
        self.thr = self.mu + AE_SIGMA * self.sd
        return self

    def _err(self, Xs):
        return np.mean((Xs - self.m.predict(Xs)) ** 2, axis=1)

    def score(self, X):
        Xs = self.sc.transform(X)
        e  = self._err(Xs)
        return e, e > self.thr


def fill_scale(df, cols, sc=None):
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


def train_cpu_models(df_tr: pd.DataFrame, df_te: pd.DataFrame,
                     out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    excl = {"instance", "timestamp", "hw_config"}
    fcols_tr = [c for c in df_tr.columns if c not in excl
                and pd.api.types.is_numeric_dtype(df_tr[c])]
    fcols_te = set(c for c in df_te.columns if c not in excl
                   and pd.api.types.is_numeric_dtype(df_te[c])) if not df_te.empty else set(fcols_tr)
    fcols = [c for c in fcols_tr if c in fcols_te]
    print(f"\n  Feature columns: {len(fcols)}")

    summary: Dict[str, Any] = {"per_farm": {}}
    all_scores: List[pd.DataFrame] = []

    for hw in sorted(df_tr["hw_config"].unique()):
        gtr = df_tr[df_tr["hw_config"] == hw]
        gte = df_te[df_te["hw_config"] == hw] if not df_te.empty else pd.DataFrame()
        n   = len(gtr)

        if n < MIN_ROWS:
            print(f"\n  [{hw}] SKIP — {n} rows < {MIN_ROWS}")
            continue

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  train={n:,}  test={len(gte):,}")

        Xtr, sc = fill_scale(gtr, fcols)
        Xte, _  = fill_scale(gte, fcols, sc) if not gte.empty else (np.array([]), sc)

        # PCA
        pca   = PCA(n_components=PCA_VAR, svd_solver="full")
        Xp_tr = pca.fit_transform(Xtr)
        Xp_te = pca.transform(Xte) if Xte.shape[0] > 0 else np.array([])
        print(f"  [{hw}] PCA: {len(fcols)} → {Xp_tr.shape[1]} components")

        # Isolation Forest
        print(f"  [{hw}] Training IsolationForest...")
        clf = IsolationForest(contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
        clf.fit(Xp_tr)
        if_tr_sc  = -clf.score_samples(Xp_tr)
        if_tr_fl  = clf.predict(Xp_tr) == -1
        if_tr_pct = if_tr_fl.mean()
        print(f"  [{hw}] IF train flagged: {if_tr_pct*100:.2f}%")

        if if_tr_pct > IF_FLAG_LIMIT:
            print(f"  [{hw}] ⚠ IF flags > 15% — skipping")
            continue

        if_te_sc = if_te_fl = None
        if Xp_te.shape[0] > 0:
            if_te_sc = -clf.score_samples(Xp_te)
            if_te_fl = clf.predict(Xp_te) == -1
            print(f"  [{hw}] IF test  flagged: {if_te_fl.mean()*100:.2f}%")

        # Autoencoder
        print(f"  [{hw}] Training Autoencoder...")
        ae = ReconAE(Xtr.shape[1])
        ae.fit(Xtr)
        ae_tr_e, ae_tr_fl = ae.score(Xtr)
        print(f"  [{hw}] AE train flagged: {ae_tr_fl.mean()*100:.2f}%  thr={ae.thr:.4f}")

        ae_te_e = ae_te_fl = None
        if Xte.shape[0] > 0:
            ae_te_e, ae_te_fl = ae.score(Xte)
            print(f"  [{hw}] AE test  flagged: {ae_te_fl.mean()*100:.2f}%")

        # Save models
        joblib.dump(sc,  out_dir / f"cpu_scaler_{hw}.joblib")
        joblib.dump(pca, out_dir / f"cpu_pca_{hw}.joblib")
        joblib.dump(clf, out_dir / f"cpu_isoforest_{hw}.joblib")
        joblib.dump(ae,  out_dir / f"cpu_autoencoder_{hw}.joblib")
        print(f"  [{hw}] ✓ Saved: scaler, pca, isoforest, autoencoder")

        # Collect scores
        meta = ["instance", "timestamp", "hw_config"]
        tr_out = gtr[meta].copy().reset_index(drop=True)
        tr_out["if_score"] = if_tr_sc; tr_out["if_flag"] = if_tr_fl
        tr_out["ae_score"] = ae_tr_e;  tr_out["ae_flag"] = ae_tr_fl
        tr_out["split"]    = "train"
        all_scores.append(tr_out)

        if if_te_fl is not None:
            te_out = gte[meta].copy().reset_index(drop=True)
            te_out["if_score"] = if_te_sc; te_out["if_flag"] = if_te_fl
            te_out["ae_score"] = ae_te_e;  te_out["ae_flag"] = ae_te_fl
            te_out["split"]    = "test"
            all_scores.append(te_out)

        summary["per_farm"][hw] = {
            "train_rows":        int(n),
            "test_rows":         int(len(gte)),
            "pca_components":    int(Xp_tr.shape[1]),
            "if_train_flag_pct": round(float(if_tr_pct) * 100, 3),
            "if_test_flag_pct":  round(float(if_te_fl.mean()) * 100, 3) if if_te_fl is not None else None,
            "ae_thr":            round(ae.thr, 6),
            "ae_train_flag_pct": round(float(ae_tr_fl.mean()) * 100, 3),
            "ae_test_flag_pct":  round(float(ae_te_fl.mean()) * 100, 3) if ae_te_fl is not None else None,
        }

    if all_scores:
        scores_df = pd.concat(all_scores, ignore_index=True)
        sp = out_dir / "cpu_anomaly_scores.parquet"
        scores_df.to_parquet(sp, index=False, compression="snappy")
        print(f"\n  ✓ Anomaly scores: {sp}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def write_json(path: Path, obj: Any) -> None:
    import json
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


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

    import datetime
    summary: Dict[str, Any] = {
        "pipeline": "cpu_data_v2",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # Step 1: Splits
    if not args.skip_splits:
        summary["splits"] = build_cpu_splits(args.clean_dir, args.splits_dir)
    else:
        print("  [SKIP] Splits")

    # Step 2: Features
    tr_feat = args.features_dir / "cpu_data_train_features.parquet"
    te_feat = args.features_dir / "cpu_data_test_features.parquet"

    if not args.skip_features:
        print(f"\n{'='*60}")
        print("  STEP 2 — CPU FEATURES")
        print(f"{'='*60}")
        tr_rows, fnames = build_cpu_features(args.splits_dir / "cpu_data_train.parquet",
                                             tr_feat, "TRAIN")
        te_rows, _      = build_cpu_features(args.splits_dir / "cpu_data_test.parquet",
                                             te_feat, "TEST")
        summary["features"] = {"train_rows": tr_rows, "test_rows": te_rows,
                                "n_features": len(fnames)}
    else:
        print("  [SKIP] Features")

    # Step 3: Train
    print(f"\n{'='*60}")
    print("  STEP 3 — CPU TRAINING")
    print(f"{'='*60}")

    if not tr_feat.exists():
        print(f"  [ERROR] {tr_feat} missing — cannot train")
        return

    df_tr = pd.read_parquet(tr_feat)
    df_te = pd.read_parquet(te_feat) if te_feat.exists() else pd.DataFrame()
    print(f"  Train features: {df_tr.shape}")
    print(f"  Test  features: {df_te.shape}")

    model_info = train_cpu_models(df_tr, df_te, args.models_dir)
    summary["models"] = model_info

    sp = args.models_dir / "cpu_training_summary.json"
    write_json(sp, summary)

    print(f"\n{'='*60}")
    print("  ✓  CPU PIPELINE v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Summary → {sp}")
    print("\n  Per-farm results:")
    for hw, info in model_info.get("per_farm", {}).items():
        ift = info.get("if_test_flag_pct")
        aet = info.get("ae_test_flag_pct")
        print(f"    {hw}: train={info['train_rows']:,}  "
              f"IF_test={ift:.1f}%  AE_test={aet:.1f}%"
              if ift is not None else
              f"    {hw}: train={info['train_rows']:,}  (no test rows)")


if __name__ == "__main__":
    main()
