"""
eda/cpu_pipeline.py
===================
Full CPU anomaly detection pipeline:
  Step 1 — Split  : read structural_clean/cpu_data/*.parquet
                    → compute per-CPU-mode rates (diff of cumulative counters)
                    → aggregate to per-node utilisation ratios
                    → chronological split (Train: May 19-22, Test: May 23)
  Step 2 — Feats  : rolling windows (5/15/60min) + lag-diff on per-node CPU utilisation
  Step 3 — Train  : IsolationForest + Autoencoder per hw_config (farm14/16/18/19/23)

Schema of cleaned CPU parquets (structural_clean/cpu_data/*.parquet):
  Columns: [__name__, instance, job, mode, timestamp, value]
  __name__  : 'node_cpu_seconds_total'
  instance  : node identifier e.g. 'farm140151:9100'
  mode      : 'user', 'system', 'idle', 'iowait', 'irq', 'softirq', 'steal', 'guest', ...
  timestamp : UTC datetime
  value     : cumulative CPU seconds (counter — must diff to get rate)
  hw_config : first 6 chars of instance → farm14/farm16/farm18/farm19/farm23

Strategy:
  1. Compute per-node per-mode rate = diff(value)/diff(time_seconds)
  2. Pivot modes → columns: cpu_user_rate, cpu_system_rate, cpu_idle_rate, cpu_iowait_rate, ...
  3. Compute utilisation ratios: user_ratio = user_rate / total_rate, etc.
  4. Binary split on time: train (May 19-22) / test (May 23)
  5. Rolling window features (5/15/60min mean/std/min/max) + lag-1 diff
  6. Train IF + AE per hw_config (same farms as memory/slurm)

Outputs:
  artifacts/splits/cpu_data_train.parquet
  artifacts/splits/cpu_data_test.parquet
  artifacts/features/cpu_data_train_features.parquet
  artifacts/features/cpu_data_test_features.parquet
  artifacts/models/cpu_isoforest_farmXX.joblib   (× 5 farms)
  artifacts/models/cpu_autoencoder_farmXX.joblib (× 5 farms)
  artifacts/models/cpu_scaler_farmXX.joblib      (× 5 farms)
  artifacts/models/cpu_training_summary.json
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from eda.lib import ensure_dir, write_json

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Config ────────────────────────────────────────────────────────────────────
TRAIN_START = pd.Timestamp("2023-05-19 00:00:00")
TRAIN_END   = pd.Timestamp("2023-05-22 23:59:59.999999999")
TEST_START  = pd.Timestamp("2023-05-23 00:00:00")
TEST_END    = pd.Timestamp("2023-05-23 23:59:59.999999999")

HW_CONFIGS = {"farm14", "farm16", "farm18", "farm19", "farm23"}

# CPU feature engineering
RESAMPLE_FREQ  = "1min"        # resample aggregated CPU data to 1-min
ROLLING_WINS   = ["5min", "15min", "60min"]
ROLLING_AGGS   = ["mean", "std", "min", "max"]
NAN_DROP_THRESH = 0.30

# CPU modes to keep (filter out guest/steal which are usually zero on bare-metal)
KEEP_MODES = {"user", "system", "idle", "iowait", "irq", "softirq"}

# Model config
IF_CONTAMINATION   = 0.05
IF_FLAG_HARD_LIMIT = 0.15
PCA_VAR_RETAINED   = 0.95
MIN_TRAIN_ROWS     = 200      # lower than memory since CPU per-node has fewer rows
AE_HIDDEN          = [64, 16]
AE_EPOCHS          = 30
AE_BATCH           = 256
AE_LR              = 1e-3
AE_SIGMA           = 3.0

DISTRESS_STATES = frozenset({
    "down", "drained", "draining", "fail", "failing",
    "down*", "drained*", "drain", "failing*",
})


# ─── Helpers ───────────────────────────────────────────────────────────────────

def hw_config(instance: str) -> str:
    clean = str(instance).split(":")[0][:6].lower()
    return clean if clean in HW_CONFIGS else "unknown"


def fill_and_scale(df: pd.DataFrame, feat_cols: List[str],
                   scaler: Optional[StandardScaler] = None
                   ) -> Tuple[np.ndarray, StandardScaler]:
    X = df[feat_cols].copy()
    for c in feat_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    X_arr = X.to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler()
        X_arr  = scaler.fit_transform(X_arr)
    else:
        X_arr  = scaler.transform(X_arr)
    return X_arr, scaler


# ─── Autoencoder ───────────────────────────────────────────────────────────────

class ReconstructionAutoencoder:
    def __init__(self, input_dim: int):
        h1 = min(64, max(input_dim, 4))
        h2 = min(16, max(input_dim // 4, 2))
        self.scaler = StandardScaler()
        self.model  = MLPRegressor(
            hidden_layer_sizes=(h1, h2, h1),
            activation="relu", solver="adam",
            learning_rate_init=AE_LR, max_iter=AE_EPOCHS,
            batch_size=AE_BATCH, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, verbose=False,
        )
        self.threshold_: float = 0.0
        self.train_mean_: float = 0.0
        self.train_std_:  float = 0.0

    def fit(self, X: np.ndarray) -> "ReconstructionAutoencoder":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, Xs)
        errs = self._errors(Xs)
        self.train_mean_ = float(np.mean(errs))
        self.train_std_  = float(np.std(errs))
        self.threshold_  = self.train_mean_ + AE_SIGMA * self.train_std_
        return self

    def _errors(self, Xs: np.ndarray) -> np.ndarray:
        return np.mean((Xs - self.model.predict(Xs)) ** 2, axis=1)

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(X)
        e  = self._errors(Xs)
        return e, e > self.threshold_


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CPU SPLITS
# ═══════════════════════════════════════════════════════════════════════════════

def build_cpu_splits(clean_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Read all cleaned CPU parquets, compute rates, aggregate per-node per-minute,
    then chronologically split into train/test.
    """
    cpu_dir = clean_dir / "cpu_data"
    files   = sorted(cpu_dir.glob("*.parquet"))
    if not files:
        print(f"[ERROR] No parquets found in {cpu_dir}")
        return {}

    print(f"\n{'='*60}")
    print(f"  CPU SPLITS  |  files: {len(files)}")
    print(f"{'='*60}")

    # ── Load all files, filter to our time range and hw_configs ────────────────
    frames: List[pd.DataFrame] = []
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            # Filter to time window we care about (May 19-23)
            mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TEST_END)
            df   = df[mask]
            if df.empty:
                continue
            df["hw_config"] = df["instance"].apply(hw_config)
            df = df[df["hw_config"] != "unknown"]
            # Filter to desired modes
            if "mode" in df.columns:
                df = df[df["mode"].isin(KEEP_MODES)]
            frames.append(df)
            if (i + 1) % 20 == 0:
                print(f"    Loaded {i+1}/{len(files)} files...")
        except Exception as e:
            print(f"    [WARN] {f.name}: {e}")

    if not frames:
        print("[ERROR] No valid CPU data found in time range")
        return {}

    print(f"  Concatenating {len(frames)} frames...")
    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded: {len(df_all):,}")
    print(f"  hw_config distribution:\n{df_all['hw_config'].value_counts().to_string()}")

    # ── Compute per-node per-mode rate (diff of cumulative counter) ─────────────
    print("\n  Computing CPU utilisation rates (diff of cumulative counters)...")

    mode_col     = "mode" if "mode" in df_all.columns else None
    instance_col = "instance"
    time_col     = "timestamp"
    value_col    = "value"

    rate_frames: List[pd.DataFrame] = []

    group_cols = [instance_col, mode_col] if mode_col else [instance_col]
    groups = df_all.groupby(group_cols, sort=False)
    total_groups = len(groups)
    processed = 0

    for key, grp in groups:
        inst = key[0] if isinstance(key, tuple) else key
        mode = key[1] if isinstance(key, tuple) and mode_col else "cpu"

        grp = grp.sort_values(time_col).copy()
        grp = grp.drop_duplicates(subset=[time_col])

        # Diff of cumulative value to get per-second rate
        dt_sec  = grp[time_col].diff().dt.total_seconds()
        dv      = grp[value_col].diff()
        # Rate = delta_value / delta_time (CPU seconds per second = CPU fraction)
        rate    = dv / dt_sec
        # Clip negative rates (counter resets)
        rate    = rate.clip(lower=0.0)

        rate_df = pd.DataFrame({
            time_col:     grp[time_col].values,
            "instance":   inst,
            "hw_config":  grp["hw_config"].iloc[0],
            "mode":       mode,
            "cpu_rate":   rate.values,
        })
        rate_frames.append(rate_df)
        processed += 1
        if processed % 500 == 0:
            pct = 100 * processed / total_groups
            print(f"    Rate computation: {processed}/{total_groups} groups ({pct:.0f}%)...")

    print(f"  Rate computation done. Aggregating...")
    df_rates = pd.concat(rate_frames, ignore_index=True).dropna(subset=["cpu_rate"])

    # ── Pivot modes → wide per-node per-timestamp ───────────────────────────────
    # Resample to 1-min grid first (some files may have different scrape intervals)
    print("  Pivoting modes to wide format...")

    node_pieces: List[pd.DataFrame] = []
    for inst, inst_grp in df_rates.groupby("instance", sort=False):
        hw = inst_grp["hw_config"].iloc[0]
        mode_pivot: Dict[str, pd.Series] = {}

        for mode, mode_grp in inst_grp.groupby("mode", sort=False):
            g = mode_grp.set_index(time_col)[["cpu_rate"]].sort_index()
            g = g[~g.index.duplicated(keep="first")]
            resampled = g.resample(RESAMPLE_FREQ).mean()
            resampled = resampled.ffill(limit=3).dropna()
            mode_pivot[f"cpu_{mode}_rate"] = resampled["cpu_rate"]

        if not mode_pivot:
            continue

        wide = pd.DataFrame(mode_pivot)
        wide.index.name = time_col

        # Compute total rate and utilisation ratios
        rate_cols = [c for c in wide.columns if c.endswith("_rate")]
        wide["cpu_total_rate"] = wide[rate_cols].sum(axis=1)
        for col in rate_cols:
            mode_name = col.replace("_rate", "_util")
            wide[mode_name] = np.where(
                wide["cpu_total_rate"] > 0,
                wide[col] / wide["cpu_total_rate"],
                np.nan
            )

        wide = wide.reset_index()
        wide["instance"]  = inst
        wide["hw_config"] = hw
        node_pieces.append(wide)

    if not node_pieces:
        print("[ERROR] No node-level CPU data after pivoting")
        return {}

    df_wide = pd.concat(node_pieces, ignore_index=True)
    print(f"  Wide CPU data: {df_wide.shape}")
    print(f"  Columns: {df_wide.columns.tolist()}")

    # ── Time-based split ─────────────────────────────────────────────────────────
    df_wide[time_col] = pd.to_datetime(df_wide[time_col])
    train_mask = (df_wide[time_col] >= TRAIN_START) & (df_wide[time_col] <= TRAIN_END)
    test_mask  = (df_wide[time_col] >= TEST_START)  & (df_wide[time_col] <= TEST_END)

    df_train = df_wide[train_mask].sort_values(time_col).reset_index(drop=True)
    df_test  = df_wide[test_mask].sort_values(time_col).reset_index(drop=True)

    print(f"\n  Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}")
    print(f"  Train hw_config:\n{df_train['hw_config'].value_counts().to_string()}")
    print(f"  Test  hw_config:\n{df_test['hw_config'].value_counts().to_string()}")

    # ── Verify no temporal overlap ────────────────────────────────────────────────
    if len(df_train) and len(df_test):
        max_train = df_train[time_col].max()
        min_test  = df_test[time_col].min()
        ok = max_train < min_test
        print(f"  max_train={max_train}  min_test={min_test}")
        print(f"  Zero temporal overlap: {'OK' if ok else 'FAIL'}")

    # ── Write splits ──────────────────────────────────────────────────────────────
    ensure_dir(out_dir)
    train_path = out_dir / "cpu_data_train.parquet"
    test_path  = out_dir / "cpu_data_test.parquet"
    df_train.to_parquet(train_path, index=False, compression="snappy")
    df_test.to_parquet(test_path,   index=False, compression="snappy")
    print(f"\n  ✓ CPU splits saved:")
    print(f"    {train_path}")
    print(f"    {test_path}")

    return {
        "train_rows":    int(len(df_train)),
        "test_rows":     int(len(df_test)),
        "train_hw_dist": df_train["hw_config"].value_counts().to_dict(),
        "test_hw_dist":  df_test["hw_config"].value_counts().to_dict(),
        "columns":       df_wide.columns.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CPU FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def build_cpu_features(df: pd.DataFrame, out_path: Path, is_train: bool) -> Tuple[int, List[str]]:
    """
    Rolling window + lag-diff features on per-node CPU utilisation.
    Operates on wide CPU split (one row per instance per timestamp).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    time_col     = "timestamp"
    instance_col = "instance"
    tag = "TRAIN" if is_train else "TEST"

    if df.empty:
        return 0, []

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Columns to use as features (rates + utilisation ratios)
    meta_cols   = {instance_col, time_col, "hw_config"}
    signal_cols = [c for c in df.columns if c not in meta_cols]

    print(f"  [{tag}] Unique nodes: {df[instance_col].nunique()}  |  Signal cols: {len(signal_cols)}")

    writer = None
    total_rows    = 0
    total_dropped = 0
    feature_names: List[str] = []

    for inst, inst_df in df.groupby(instance_col, sort=False):
        hw = inst_df["hw_config"].iloc[0]
        inst_df = inst_df.set_index(time_col).sort_index()
        inst_df = inst_df[~inst_df.index.duplicated(keep="first")]

        feat_cols: Dict[str, pd.Series] = {}

        for sig_col in signal_cols:
            if sig_col not in inst_df.columns:
                continue
            series = inst_df[sig_col]
            safe   = sig_col.replace("/", "_").replace("-", "_")

            for win in ROLLING_WINS:
                roller = series.rolling(window=win, min_periods=1)
                w_safe = win.replace("min", "m")
                for agg in ROLLING_AGGS:
                    name = f"{safe}_{w_safe}_{agg}"
                    if agg == "mean":
                        feat_cols[name] = roller.mean()
                    elif agg == "std":
                        feat_cols[name] = roller.std().fillna(0.0)
                    elif agg == "min":
                        feat_cols[name] = roller.min()
                    elif agg == "max":
                        feat_cols[name] = roller.max()

            feat_cols[f"{safe}_lag1_diff"] = series.diff(1)

        if not feat_cols:
            continue

        wide = pd.DataFrame(feat_cols, index=inst_df.index).reset_index()
        wide[instance_col] = inst
        wide["hw_config"]  = hw

        # Drop rows with >30% NaN
        fn = [c for c in wide.columns if c not in [instance_col, time_col, "hw_config"]]
        n_before  = len(wide)
        nan_frac  = wide[fn].isna().mean(axis=1)
        wide      = wide[nan_frac <= NAN_DROP_THRESH].reset_index(drop=True)
        total_dropped += (n_before - len(wide))
        total_rows    += len(wide)

        if wide.empty:
            continue
        if not feature_names:
            feature_names = fn

        table = pa.Table.from_pandas(wide, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        writer.write_table(table)

    if writer:
        writer.close()

    print(f"  [{tag}] Output: {total_rows:,} rows × {len(feature_names)+3} cols  |  NaN-dropped: {total_dropped:,}")
    return total_rows, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CPU MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_feat_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_cpu_models(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     out_dir: Path) -> Dict[str, Any]:
    """
    Train IsolationForest + Autoencoder per hw_config farm.
    Saves .joblib files and anomaly score parquets.
    """
    ensure_dir(out_dir)
    summary: Dict[str, Any] = {"per_farm": {}}

    exclude_cols = ["instance", "timestamp", "hw_config"]
    feat_cols    = get_feat_cols(df_train, exclude_cols)
    feat_cols    = [c for c in feat_cols if c in (
        set(get_feat_cols(df_test, exclude_cols)) if not df_test.empty else set(feat_cols)
    )]

    print(f"\n  Feature columns: {len(feat_cols)}")

    hw_configs = df_train["hw_config"].unique()
    print(f"  hw_configs to train: {sorted(hw_configs.tolist())}")

    all_scores_train: List[pd.DataFrame] = []
    all_scores_test:  List[pd.DataFrame] = []

    for hw in sorted(hw_configs):
        mask_tr = df_train["hw_config"] == hw
        grp_tr  = df_train[mask_tr]
        grp_te  = df_test[df_test["hw_config"] == hw] if not df_test.empty else pd.DataFrame()

        n_train = len(grp_tr)
        if n_train < MIN_TRAIN_ROWS:
            print(f"\n  [{hw}] SKIP — {n_train} train rows < {MIN_TRAIN_ROWS}")
            continue

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  train={n_train:,}  test={len(grp_te):,}")

        X_tr, scaler = fill_and_scale(grp_tr, feat_cols)
        X_te, _      = fill_and_scale(grp_te, feat_cols, scaler) if not grp_te.empty else (np.array([]), scaler)

        # ── PCA for IF ─────────────────────────────────────────────────────────
        n_comp   = min(X_tr.shape[1] - 1, X_tr.shape[0] - 1)
        pca      = PCA(n_components=PCA_VAR_RETAINED, svd_solver="full")
        X_tr_pca = pca.fit_transform(X_tr)
        X_te_pca = pca.transform(X_te) if X_te.shape[0] > 0 else np.array([])
        print(f"  [{hw}] PCA: {len(feat_cols)} → {X_tr_pca.shape[1]} components")

        # ── Isolation Forest ───────────────────────────────────────────────────
        print(f"  [{hw}] Training IsolationForest...")
        clf = IsolationForest(contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
        clf.fit(X_tr_pca)

        tr_scores_if = -clf.score_samples(X_tr_pca)
        tr_flags_if  = clf.predict(X_tr_pca) == -1
        tr_flag_pct  = tr_flags_if.mean()
        print(f"  [{hw}] IF train flagged: {tr_flag_pct*100:.2f}%")

        if tr_flag_pct > IF_FLAG_HARD_LIMIT:
            print(f"  [ERROR] [{hw}] IF flags {tr_flag_pct*100:.1f}% > 15% limit — skipping farm")
            continue

        te_scores_if = te_flags_if = None
        if X_te_pca.shape[0] > 0:
            te_scores_if = -clf.score_samples(X_te_pca)
            te_flags_if  = clf.predict(X_te_pca) == -1
            print(f"  [{hw}] IF test  flagged: {te_flags_if.mean()*100:.2f}%")

        # ── Autoencoder ────────────────────────────────────────────────────────
        print(f"  [{hw}] Training Autoencoder...")
        ae = ReconstructionAutoencoder(input_dim=X_tr.shape[1])
        ae.fit(X_tr)
        tr_err_ae, tr_flags_ae = ae.score(X_tr)
        print(f"  [{hw}] AE train flagged: {tr_flags_ae.mean()*100:.2f}%  threshold={ae.threshold_:.6f}")

        te_err_ae = te_flags_ae = None
        if X_te.shape[0] > 0:
            te_err_ae, te_flags_ae = ae.score(X_te)
            print(f"  [{hw}] AE test  flagged: {te_flags_ae.mean()*100:.2f}%")

        # ── Save models ────────────────────────────────────────────────────────
        joblib.dump(scaler, out_dir / f"cpu_scaler_{hw}.joblib")
        joblib.dump(pca,    out_dir / f"cpu_pca_{hw}.joblib")
        joblib.dump(clf,    out_dir / f"cpu_isoforest_{hw}.joblib")
        joblib.dump(ae,     out_dir / f"cpu_autoencoder_{hw}.joblib")
        print(f"  [{hw}] ✓ Models saved")

        # ── Collect scores ─────────────────────────────────────────────────────
        tr_out = grp_tr[["instance", "timestamp", "hw_config"]].copy().reset_index(drop=True)
        tr_out["if_score"]   = tr_scores_if
        tr_out["if_flag"]    = tr_flags_if
        tr_out["ae_score"]   = tr_err_ae
        tr_out["ae_flag"]    = tr_flags_ae
        tr_out["split"]      = "train"
        all_scores_train.append(tr_out)

        if te_scores_if is not None and X_te_pca.shape[0] > 0:
            te_out = grp_te[["instance", "timestamp", "hw_config"]].copy().reset_index(drop=True)
            te_out["if_score"] = te_scores_if
            te_out["if_flag"]  = te_flags_if
            te_out["ae_score"] = te_err_ae
            te_out["ae_flag"]  = te_flags_ae
            te_out["split"]    = "test"
            all_scores_test.append(te_out)

        summary["per_farm"][hw] = {
            "train_rows":        int(n_train),
            "test_rows":         int(len(grp_te)),
            "pca_components":    int(X_tr_pca.shape[1]),
            "if_train_flag_pct": round(float(tr_flag_pct) * 100, 3),
            "if_test_flag_pct":  round(float(te_flags_if.mean()) * 100, 3) if te_flags_if is not None else None,
            "ae_threshold":      round(ae.threshold_, 6),
            "ae_train_flag_pct": round(float(tr_flags_ae.mean()) * 100, 3),
            "ae_test_flag_pct":  round(float(te_flags_ae.mean()) * 100, 3) if te_flags_ae is not None else None,
        }

    # ── Save combined scores ───────────────────────────────────────────────────
    if all_scores_train:
        all_scores = pd.concat(all_scores_train + all_scores_test, ignore_index=True)
        scores_path = out_dir / "cpu_anomaly_scores.parquet"
        all_scores.to_parquet(scores_path, index=False, compression="snappy")
        print(f"\n  ✓ Scores saved: {scores_path}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Full CPU anomaly detection pipeline: splits → features → train")
    ap.add_argument("--clean-dir",     type=Path, default=Path("structural_clean"),
                    help="Directory containing cpu_data/ cleaned parquets")
    ap.add_argument("--splits-dir",    type=Path, default=Path("artifacts/splits"),
                    help="Output directory for CPU split parquets")
    ap.add_argument("--features-dir",  type=Path, default=Path("artifacts/features"),
                    help="Output directory for CPU feature parquets")
    ap.add_argument("--models-dir",    type=Path, default=Path("artifacts/models"),
                    help="Output directory for trained models + scores")
    ap.add_argument("--skip-splits",   action="store_true",
                    help="Skip split step (use existing cpu_data_*.parquet)")
    ap.add_argument("--skip-features", action="store_true",
                    help="Skip feature step (use existing cpu_data_*_features.parquet)")
    args = ap.parse_args()

    ensure_dir(args.splits_dir)
    ensure_dir(args.features_dir)
    ensure_dir(args.models_dir)

    summary: Dict[str, Any] = {
        "pipeline": "cpu_data",
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }

    # ── Step 1: Splits ─────────────────────────────────────────────────────────
    if not args.skip_splits:
        print("\n" + "="*60)
        print("  STEP 1 — CPU SPLITS")
        print("="*60)
        split_info = build_cpu_splits(args.clean_dir, args.splits_dir)
        summary["splits"] = split_info
    else:
        print("  [SKIP] Splits — using existing parquets")

    # ── Step 2: Features ───────────────────────────────────────────────────────
    train_feat_path = args.features_dir / "cpu_data_train_features.parquet"
    test_feat_path  = args.features_dir / "cpu_data_test_features.parquet"

    if not args.skip_features:
        print("\n" + "="*60)
        print("  STEP 2 — CPU FEATURE ENGINEERING")
        print("="*60)

        train_split = args.splits_dir / "cpu_data_train.parquet"
        test_split  = args.splits_dir / "cpu_data_test.parquet"

        if not train_split.exists():
            print(f"  [ERROR] {train_split} not found. Run without --skip-splits.")
            return

        df_tr = pd.read_parquet(train_split)
        df_te = pd.read_parquet(test_split) if test_split.exists() else pd.DataFrame()
        print(f"  Train split: {df_tr.shape}  |  Test split: {df_te.shape}")

        tr_rows, feat_names = build_cpu_features(df_tr, train_feat_path, is_train=True)
        te_rows, _          = build_cpu_features(df_te, test_feat_path,  is_train=False) if not df_te.empty else (0, [])

        summary["features"] = {
            "train_rows":   tr_rows,
            "test_rows":    te_rows,
            "n_features":   len(feat_names),
            "feature_names": feat_names[:20],   # truncated for readability
        }
    else:
        print("  [SKIP] Features — using existing parquets")

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — CPU MODEL TRAINING")
    print("="*60)

    if not train_feat_path.exists():
        print(f"  [ERROR] {train_feat_path} not found. Cannot train.")
        return

    df_train_feats = pd.read_parquet(train_feat_path)
    df_test_feats  = pd.read_parquet(test_feat_path) if test_feat_path.exists() else pd.DataFrame()
    print(f"  Train features: {df_train_feats.shape}")
    print(f"  Test  features: {df_test_feats.shape}")

    model_summary = train_cpu_models(df_train_feats, df_test_feats, args.models_dir)
    summary["models"] = model_summary

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_path = args.models_dir / "cpu_training_summary.json"
    write_json(summary_path, summary)
    print(f"\n{'='*60}")
    print(f"  ✓ CPU PIPELINE COMPLETE")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")

    print("\n  Per-farm results:")
    for hw, info in model_summary.get("per_farm", {}).items():
        if_pct = info.get("if_test_flag_pct")
        ae_pct = info.get("ae_test_flag_pct")
        print(f"    {hw}: train={info['train_rows']:,}  "
              f"IF_test={if_pct:.1f}%  AE_test={ae_pct:.1f}%" if if_pct is not None else
              f"    {hw}: train={info['train_rows']:,}  (no test data)")


if __name__ == "__main__":
    main()
