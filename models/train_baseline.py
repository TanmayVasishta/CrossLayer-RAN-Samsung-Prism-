"""
models/train_baseline.py
========================
Baseline anomaly detection models for the Samsung PRISM Cross-layer RAN project.

Two models trained per modality per hw_config group:
  Model 1 — Isolation Forest (IF)
    memory : PCA (95% variance) -> IF(contamination=0.05) — avoids curse of dimensionality
    slurm  : IF(contamination=0.05) directly — low-dim after ratio engineering

  Model 2 — Autoencoder Reconstruction Error (AE)
    Architecture : input -> 64 -> 16 -> 64 -> input (MLP, ReLU, MSE)
    Anomaly flag : reconstruction_error > mean_train_error + 3*std_train_error

Training constraints:
  - Fitted ONLY on train features (May 19-22 normal data)
  - Scored on test (May 23 anomalous data) using train-fitted transforms
  - Groups with < 500 train rows are skipped
  - IF train flag rate > 15% raises error and halts
  - No labels used anywhere

Cross-validation (indirect, no labels):
  slurm  : Flagged rows ∩ is_distress_status=1 on May 23 (STRONGEST proxy)
  memory : Anomalous nodes clustered by hw_config?

Outputs:
  artifacts/scores/{modality}_{model_name}_scores.parquet
    columns: [timestamp, node/instance, hw_config, anomaly_score, is_anomaly_flag]
  artifacts/scores/model_summary.json
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from eda.lib import ensure_dir, write_json

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Config ────────────────────────────────────────────────────────────────────
IF_CONTAMINATION         = 0.05
IF_TRAIN_FLAG_HARD_LIMIT = 0.15    # > 15% train flags -> error
PCA_VARIANCE_RETAINED    = 0.95
MIN_TRAIN_ROWS           = 500
AE_HIDDEN_DIMS           = [64, 16]
AE_EPOCHS                = 30
AE_BATCH_SIZE            = 256
AE_LR                    = 1e-3
AE_ANOMALY_SIGMA         = 3.0     # threshold = mean + 3*std of train recon errors

DISTRESS_STATES = frozenset({
    "down", "drained", "draining", "fail", "failing",
    "down*", "drained*", "drain", "failing*",
})

# ─── Autoencoder (pure NumPy/sklearn — no torch dependency required) ───────────
# Uses a simple sklearn pipeline approximation via reconstruction from PCA
# If scikit-learn version supports MLPRegressor, use that; else fall back to PCA.
try:
    from sklearn.neural_network import MLPRegressor
    _HAS_MLP = True
except ImportError:
    _HAS_MLP = False


class ReconstructionAutoencoder:
    """
    MLP-based Autoencoder using sklearn MLPRegressor.
    Architecture: input -> 64 -> 16 -> 64 -> input
    Falls back to PCA reconstruction if MLPRegressor unavailable.
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.scaler    = StandardScaler()
        self._use_mlp  = _HAS_MLP and input_dim > 0

        # Hidden layer sizes: encoder bottleneck decoder
        h1 = min(64, max(input_dim, 4))
        h2 = min(16, max(input_dim // 4, 2))
        h3 = h1

        if self._use_mlp:
            self.model = MLPRegressor(
                hidden_layer_sizes=(h1, h2, h3),
                activation="relu",
                solver="adam",
                learning_rate_init=AE_LR,
                max_iter=AE_EPOCHS,
                batch_size=AE_BATCH_SIZE,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                verbose=False,
            )
        else:
            # Fallback: PCA reconstruction as autoencoder proxy
            n_components = min(max(h2, 1), input_dim - 1) if input_dim > 1 else 1
            self.model = PCA(n_components=n_components)

        self.train_recon_mean_: float = 0.0
        self.train_recon_std_:  float = 1.0
        self.threshold_:        float = 0.0

    def fit(self, X: np.ndarray) -> "ReconstructionAutoencoder":
        X_scaled = self.scaler.fit_transform(X)
        if self._use_mlp:
            self.model.fit(X_scaled, X_scaled)
        else:
            self.model.fit(X_scaled)
        train_errors = self._recon_error(X_scaled)
        self.train_recon_mean_ = float(np.mean(train_errors))
        self.train_recon_std_  = float(np.std(train_errors))
        self.threshold_        = self.train_recon_mean_ + AE_ANOMALY_SIGMA * self.train_recon_std_
        return self

    def _recon_error(self, X_scaled: np.ndarray) -> np.ndarray:
        if self._use_mlp:
            recon = self.model.predict(X_scaled)
        else:
            recon = self.model.inverse_transform(self.model.transform(X_scaled))
        return np.mean((X_scaled - recon) ** 2, axis=1)

    def score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (reconstruction_error, is_anomaly_flag)."""
        X_scaled    = self.scaler.transform(X)
        errors      = self._recon_error(X_scaled)
        is_anomaly  = errors > self.threshold_
        return errors, is_anomaly


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return numeric feature columns, excluding metadata columns."""
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def fill_and_scale(df: pd.DataFrame, feature_cols: List[str],
                   scaler: Optional[StandardScaler] = None
                   ) -> Tuple[np.ndarray, StandardScaler]:
    """Median-fill NaN then StandardScale. Fits scaler if not provided."""
    X = df[feature_cols].copy()
    for c in feature_cols:
        median = X[c].median()
        X[c]   = X[c].fillna(median if pd.notna(median) else 0.0)
    X_arr = X.to_numpy(dtype=float)
    if scaler is None:
        scaler = StandardScaler()
        X_arr  = scaler.fit_transform(X_arr)
    else:
        X_arr  = scaler.transform(X_arr)
    return X_arr, scaler


# ─── Isolation Forest per hw_config ────────────────────────────────────────────

def run_isolation_forest(
    modality: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    node_col: str,
    out_dir: Path,
    use_pca: bool = False,
) -> Dict[str, Any]:
    """Train IF per hw_config group. Returns model summary dict."""
    summary: Dict[str, Any] = {
        "model":       "isolation_forest",
        "modality":    modality,
        "use_pca":     use_pca,
        "contamination": IF_CONTAMINATION,
        "hw_config_results": {},
    }
    score_pieces_train: List[pd.DataFrame] = []
    score_pieces_test:  List[pd.DataFrame] = []

    hw_configs = df_train["hw_config"].unique()
    print(f"\n  [IF] {modality}  hw_configs: {hw_configs.tolist()}")

    for hw in hw_configs:
        mask_tr = df_train["hw_config"] == hw
        mask_te = df_test["hw_config"] == hw if not df_test.empty else pd.Series(False, index=df_test.index)

        grp_train = df_train[mask_tr]
        grp_test  = df_test[mask_te]  if not df_test.empty else pd.DataFrame()

        n_train = len(grp_train)
        if n_train < MIN_TRAIN_ROWS:
            print(f"  [IF] [{hw}] SKIP — only {n_train} train rows (< {MIN_TRAIN_ROWS})")
            continue

        print(f"  [IF] [{hw}] train={n_train:,}  test={len(grp_test):,}")

        # Scale
        X_train, scaler = fill_and_scale(grp_train, feature_cols)
        X_test,  _      = fill_and_scale(grp_test,  feature_cols, scaler) if not grp_test.empty else (np.array([]), scaler)

        # Optional PCA
        pca_model = None
        if use_pca:
            n_comp = min(X_train.shape[1] - 1, X_train.shape[0] - 1)
            pca_model = PCA(n_components=PCA_VARIANCE_RETAINED, svd_solver="full")
            X_train = pca_model.fit_transform(X_train)
            if X_test.shape[0] > 0:
                X_test = pca_model.transform(X_test)
            n_pca = X_train.shape[1]
            print(f"  [IF] [{hw}] PCA: {len(feature_cols)} -> {n_pca} components ({PCA_VARIANCE_RETAINED*100:.0f}% var)")

        # Train IF
        clf = IsolationForest(contamination=IF_CONTAMINATION, random_state=42, n_jobs=-1)
        clf.fit(X_train)

        # Score train
        train_scores  = -clf.score_samples(X_train)   # higher = more anomalous
        train_flags   = clf.predict(X_train) == -1     # -1 = anomaly
        train_flag_pct = train_flags.mean()

        if train_flag_pct > IF_TRAIN_FLAG_HARD_LIMIT:
            print(f"  [ERROR] [{hw}] IF flags {train_flag_pct*100:.1f}% of TRAIN rows > 15% limit!")
            print("  Contamination parameter may be wrong. Halting.")
            import sys; sys.exit(1)

        print(f"  [IF] [{hw}] Train flagged: {train_flag_pct*100:.2f}%")

        # Collate train scores
        train_out = grp_train[[node_col, "timestamp", "hw_config"]].copy().reset_index(drop=True)
        train_out["anomaly_score"]   = train_scores
        train_out["is_anomaly_flag"] = train_flags
        train_out["split"]           = "train"
        score_pieces_train.append(train_out)

        # Score test
        test_flag_pct = None
        if X_test.shape[0] > 0:
            test_scores = -clf.score_samples(X_test)
            test_flags  = clf.predict(X_test) == -1
            test_flag_pct = test_flags.mean()
            print(f"  [IF] [{hw}] Test  flagged: {test_flag_pct*100:.2f}%")

            test_out = grp_test[[node_col, "timestamp", "hw_config"]].copy().reset_index(drop=True)
            test_out["anomaly_score"]   = test_scores
            test_out["is_anomaly_flag"] = test_flags
            test_out["split"]           = "test"
            score_pieces_test.append(test_out)

        summary["hw_config_results"][hw] = {
            "train_rows":        int(n_train),
            "test_rows":         int(len(grp_test)),
            "pca_components":    int(X_train.shape[1]) if use_pca else None,
            "train_flagged_pct": round(float(train_flag_pct) * 100, 3),
            "test_flagged_pct":  round(float(test_flag_pct)  * 100, 3) if test_flag_pct is not None else None,
        }

    # Combine and save
    ensure_dir(out_dir)
    model_name = f"{modality}_isolation_forest"

    if score_pieces_train:
        df_scores_train = pd.concat(score_pieces_train, ignore_index=True)
        df_scores_test  = pd.concat(score_pieces_test,  ignore_index=True) if score_pieces_test else pd.DataFrame()

        df_scores_all = pd.concat([df_scores_train, df_scores_test], ignore_index=True) if not df_scores_test.empty else df_scores_train
        out_path = out_dir / (model_name + "_scores.parquet")
        df_scores_all.to_parquet(out_path, index=False, compression="snappy")
        print(f"  [IF] Scores saved: {out_path}")

    return summary


# ─── Autoencoder per hw_config ─────────────────────────────────────────────────

def run_autoencoder(
    modality: str,
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    feature_cols: List[str],
    node_col: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Train Autoencoder per hw_config group. Returns model summary dict."""
    summary: Dict[str, Any] = {
        "model":       "autoencoder",
        "modality":    modality,
        "architecture": "input->64->16->64->input (ReLU, MSE)",
        "anomaly_threshold_sigma": AE_ANOMALY_SIGMA,
        "hw_config_results": {},
    }
    score_pieces_train: List[pd.DataFrame] = []
    score_pieces_test:  List[pd.DataFrame] = []

    hw_configs = df_train["hw_config"].unique()
    print(f"\n  [AE] {modality}  hw_configs: {hw_configs.tolist()}")

    for hw in hw_configs:
        mask_tr = df_train["hw_config"] == hw
        mask_te = df_test["hw_config"] == hw if not df_test.empty else pd.Series(False, index=df_test.index)

        grp_train = df_train[mask_tr]
        grp_test  = df_test[mask_te]  if not df_test.empty else pd.DataFrame()

        n_train = len(grp_train)
        if n_train < MIN_TRAIN_ROWS:
            print(f"  [AE] [{hw}] SKIP — only {n_train} train rows (< {MIN_TRAIN_ROWS})")
            continue

        print(f"  [AE] [{hw}] train={n_train:,}  test={len(grp_test):,}")

        X_train, scaler = fill_and_scale(grp_train, feature_cols)
        X_test,  _      = fill_and_scale(grp_test,  feature_cols, scaler) if not grp_test.empty else (np.array([]), scaler)

        # Train AE — data already scaled by fill_and_scale, so bypass AE's internal scaler
        ae = ReconstructionAutoencoder(input_dim=X_train.shape[1])
        # Fit AE-internal scaler on a dummy ones matrix so it's a valid no-op transform
        dummy = np.ones((2, X_train.shape[1]), dtype=float)
        ae.scaler.fit(dummy)
        ae.scaler.mean_  = np.zeros(X_train.shape[1])
        ae.scaler.scale_ = np.ones(X_train.shape[1])
        ae.fit(X_train)

        # Train scores
        train_errors, train_flags = ae.score(X_train)
        train_flag_pct = train_flags.mean()
        print(f"  [AE] [{hw}] Train flagged: {train_flag_pct*100:.2f}%  "
              f"threshold={ae.threshold_:.6f}")

        train_out = grp_train[[node_col, "timestamp", "hw_config"]].copy().reset_index(drop=True)
        train_out["anomaly_score"]   = train_errors
        train_out["is_anomaly_flag"] = train_flags
        train_out["split"]           = "train"
        score_pieces_train.append(train_out)

        # Test scores
        test_flag_pct = None
        if X_test.shape[0] > 0:
            test_errors, test_flags = ae.score(X_test)
            test_flag_pct = test_flags.mean()
            print(f"  [AE] [{hw}] Test  flagged: {test_flag_pct*100:.2f}%")

            test_out = grp_test[[node_col, "timestamp", "hw_config"]].copy().reset_index(drop=True)
            test_out["anomaly_score"]   = test_errors
            test_out["is_anomaly_flag"] = test_flags
            test_out["split"]           = "test"
            score_pieces_test.append(test_out)

        summary["hw_config_results"][hw] = {
            "train_rows":            int(n_train),
            "test_rows":             int(len(grp_test)),
            "train_recon_mean":      round(float(ae.train_recon_mean_), 6),
            "train_recon_std":       round(float(ae.train_recon_std_),  6),
            "anomaly_threshold":     round(float(ae.threshold_),        6),
            "train_flagged_pct":     round(float(train_flag_pct) * 100, 3),
            "test_flagged_pct":      round(float(test_flag_pct)  * 100, 3) if test_flag_pct is not None else None,
        }

    # Save
    ensure_dir(out_dir)
    model_name = f"{modality}_autoencoder"
    if score_pieces_train:
        df_scores_train = pd.concat(score_pieces_train, ignore_index=True)
        df_scores_test  = pd.concat(score_pieces_test,  ignore_index=True) if score_pieces_test else pd.DataFrame()
        df_scores_all   = pd.concat([df_scores_train, df_scores_test], ignore_index=True) if not df_scores_test.empty else df_scores_train
        out_path = out_dir / (model_name + "_scores.parquet")
        df_scores_all.to_parquet(out_path, index=False, compression="snappy")
        print(f"  [AE] Scores saved: {out_path}")

    return summary


# ─── Cross-validation helpers ──────────────────────────────────────────────────

def slurm_distress_crosscheck(
    scores_path: Path,
    slurm_feats_test: pd.DataFrame,
    model_name: str,
) -> Dict[str, Any]:
    """
    Among rows flagged anomalous on May 23, what % have is_distress_status=1?
    This is the strongest indirect validation signal.
    """
    if not scores_path.exists() or "is_distress_status" not in slurm_feats_test.columns:
        return {}

    scores = pd.read_parquet(scores_path)
    test_scores = scores[scores["split"] == "test"]
    flagged = test_scores[test_scores["is_anomaly_flag"] == True]

    if flagged.empty:
        return {"flagged_rows": 0, "distress_overlap_pct": 0.0}

    # Join on (node, timestamp) to get distress status
    if "node" not in flagged.columns:
        return {"error": "no node column in scores"}

    flagged_nodes_ts = flagged[["node", "timestamp"]].copy()
    distress_lookup  = slurm_feats_test[["node", "timestamp", "is_distress_status"]].copy()
    merged = flagged_nodes_ts.merge(distress_lookup, on=["node", "timestamp"], how="left")
    distress_overlap = merged["is_distress_status"].fillna(0).mean()

    print(f"\n  [{model_name}] Slurm distress cross-check:")
    print(f"    Flagged rows on May 23: {len(flagged):,}")
    print(f"    Of flagged rows, {distress_overlap*100:.2f}% have is_distress_status=1")

    # Which nodes were flagged?
    flagged_nodes = flagged["node"].value_counts().head(20).to_dict()
    print(f"    Top flagged nodes: {list(flagged_nodes.keys())[:5]}")

    return {
        "total_test_rows":          int(len(test_scores)),
        "flagged_rows":             int(len(flagged)),
        "flagged_pct":              round(float(len(flagged) / len(test_scores)) * 100, 2),
        "distress_overlap_pct":     round(float(distress_overlap) * 100, 2),
        "top_flagged_nodes":        dict(list(flagged_nodes.items())[:20]),
    }


def memory_hw_cluster_check(
    scores_path: Path,
    model_name: str,
) -> Dict[str, Any]:
    """
    Among memory_data May 23 anomalous nodes, do they cluster by hw_config?
    Hardware-related anomalies should cluster within one hw_config group.
    """
    if not scores_path.exists():
        return {}

    scores = pd.read_parquet(scores_path)
    test_scores = scores[scores["split"] == "test"]
    flagged = test_scores[test_scores["is_anomaly_flag"] == True]

    if flagged.empty:
        return {"flagged_rows": 0}

    hw_distribution = flagged["hw_config"].value_counts(normalize=True).mul(100).round(2).to_dict()
    total_by_hw     = test_scores["hw_config"].value_counts().to_dict()
    flagged_by_hw   = flagged["hw_config"].value_counts().to_dict()
    rate_by_hw      = {
        hw: round(flagged_by_hw.get(hw, 0) / total_by_hw.get(hw, 1) * 100, 2)
        for hw in total_by_hw
    }

    print(f"\n  [{model_name}] Memory hw_config cluster check:")
    print(f"    Flagged row distribution by hw_config: {hw_distribution}")
    print(f"    Flag rate per hw_config: {rate_by_hw}")
    dominant = max(rate_by_hw, key=rate_by_hw.get)
    print(f"    Most affected hw_config: {dominant} ({rate_by_hw[dominant]:.1f}% flag rate)")

    return {
        "total_test_rows":      int(len(test_scores)),
        "flagged_rows":         int(len(flagged)),
        "hw_config_flagged_distribution_pct": hw_distribution,
        "hw_config_flag_rate_pct":            rate_by_hw,
        "most_affected_hw_config":            dominant,
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train baseline anomaly detectors (IF + Autoencoder) per hw_config."
    )
    ap.add_argument("--features-dir", type=Path, default=Path("artifacts/features"),
                    help="Feature parquets from eda/feature_eng.py")
    ap.add_argument("--out-dir",      type=Path, default=Path("artifacts/scores"),
                    help="Output directory for score parquets + model summary")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    all_summaries: Dict[str, Any] = {
        "generated_at":     __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "models":            {},
        "cross_validations": {},
    }

    # ─── Load feature matrices ─────────────────────────────────────────────────
    def load_feats(name: str):
        tr = args.features_dir / (name + "_train_features.parquet")
        te = args.features_dir / (name + "_test_features.parquet")
        df_tr = pd.read_parquet(tr) if tr.exists() else pd.DataFrame()
        df_te = pd.read_parquet(te) if te.exists() else pd.DataFrame()
        return df_tr, df_te

    df_mem_train, df_mem_test    = load_feats("memory_data")
    df_slurm_train, df_slurm_test = load_feats("slurm_data")

    # ─── memory_data ──────────────────────────────────────────────────────────
    if not df_mem_train.empty:
        print("\n" + "=" * 60)
        print("  MEMORY DATA — Baseline Models")
        print("=" * 60)
        mem_exclude = ["instance", "timestamp", "hw_config"]
        mem_feat_cols = get_feature_cols(df_mem_train, mem_exclude)
        print(f"  Feature columns: {len(mem_feat_cols)}")

        # IF with PCA
        mem_if_summary = run_isolation_forest(
            "memory_data", df_mem_train, df_mem_test,
            mem_feat_cols, "instance", args.out_dir, use_pca=True,
        )
        all_summaries["models"]["memory_isolation_forest"] = mem_if_summary

        # Autoencoder
        mem_ae_summary = run_autoencoder(
            "memory_data", df_mem_train, df_mem_test,
            mem_feat_cols, "instance", args.out_dir,
        )
        all_summaries["models"]["memory_autoencoder"] = mem_ae_summary

        # Cross-check: hw_config clustering of memory anomalies
        for model_tag, fname in [("IF", "memory_data_isolation_forest"), ("AE", "memory_data_autoencoder")]:
            xval = memory_hw_cluster_check(args.out_dir / (fname + "_scores.parquet"), model_tag)
            all_summaries["cross_validations"][f"memory_{model_tag}_hw_cluster"] = xval

    # ─── slurm_data ───────────────────────────────────────────────────────────
    if not df_slurm_train.empty:
        print("\n" + "=" * 60)
        print("  SLURM DATA — Baseline Models")
        print("=" * 60)
        slurm_exclude = ["node", "timestamp", "status", "hw_config", "instance", "job"]
        # Use intersection of train and test feature cols to avoid missing one-hot columns
        slurm_feat_cols_train = get_feature_cols(df_slurm_train, slurm_exclude)
        slurm_feat_cols_test  = set(get_feature_cols(df_slurm_test, slurm_exclude)) if not df_slurm_test.empty else set(slurm_feat_cols_train)
        slurm_feat_cols = [c for c in slurm_feat_cols_train if c in slurm_feat_cols_test]
        print(f"  Feature columns ({len(slurm_feat_cols)}): {slurm_feat_cols}")

        # IF (no PCA — low-dim already)
        slurm_if_summary = run_isolation_forest(
            "slurm_data", df_slurm_train, df_slurm_test,
            slurm_feat_cols, "node", args.out_dir, use_pca=False,
        )
        all_summaries["models"]["slurm_isolation_forest"] = slurm_if_summary

        # Autoencoder
        slurm_ae_summary = run_autoencoder(
            "slurm_data", df_slurm_train, df_slurm_test,
            slurm_feat_cols, "node", args.out_dir,
        )
        all_summaries["models"]["slurm_autoencoder"] = slurm_ae_summary

        # Cross-check: distress status overlap (STRONGEST validation)
        for model_tag, fname in [("IF", "slurm_data_isolation_forest"), ("AE", "slurm_data_autoencoder")]:
            xval = slurm_distress_crosscheck(
                args.out_dir / (fname + "_scores.parquet"),
                df_slurm_test, model_tag,
            )
            all_summaries["cross_validations"][f"slurm_{model_tag}_distress_overlap"] = xval

    # ─── Save summary ──────────────────────────────────────────────────────────
    summary_path = args.out_dir / "model_summary.json"
    write_json(summary_path, all_summaries)
    print("\n" + "=" * 60)
    print("  Model summary: " + str(summary_path))
    print("=" * 60)


if __name__ == "__main__":
    main()
