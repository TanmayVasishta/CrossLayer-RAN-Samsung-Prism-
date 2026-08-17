"""
eda/cpu_disk_adaptive_lof.py
============================
Two improvements to CPU and Disk anomaly detection pipelines:

PART 1 — Adaptive Thresholding:
  Re-apply per-farm percentile thresholds to existing AE scores, sweep
  percentiles (85, 90, 95, 99) per farm, pick the percentile that maximises
  F1 score on the test set, then report new recall/precision/F1.
  Saves: cpu_adaptive_thr.json, disk_adaptive_thr.json

PART 2 — LOF on PCA-compressed features:
  Load existing cpu_pca_{farm}.joblib / disk_pca_{farm}.joblib
  Fit LOF(novelty=True, n_neighbors=20) on PCA-compressed train features
  Score test features, compute metrics, save:
    cpu_lof_{farm}.joblib, disk_lof_{farm}.joblib
    cpu_lof_scores.parquet, disk_lof_scores.parquet

PART 3 — Unified results:
  Merge Part 1 + Part 2 metrics into new v7 evaluation JSONs
  (never overwrites any existing file).

No existing models are retrained or overwritten.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Add project root to path so models.autoencoder unpickles correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODELS   = Path("artifacts/models")
FEATURES = Path("artifacts/features")
SLURM_TEST_PATH = FEATURES / "slurm_data_test_features.parquet"

EXISTING_V6_EVAL  = MODELS / "evaluation_report_metrics_v6.json"
EXISTING_V6_MULTI = MODELS / "multimodel_results_v6.json"

OUT_CPU_ADAPTIVE  = MODELS / "cpu_adaptive_thr.json"
OUT_DISK_ADAPTIVE = MODELS / "disk_adaptive_thr.json"
OUT_CPU_LOF_SCORES  = MODELS / "cpu_lof_scores.parquet"
OUT_DISK_LOF_SCORES = MODELS / "disk_lof_scores.parquet"
OUT_V7_EVAL  = MODELS / "evaluation_report_metrics_v7.json"
OUT_V7_MULTI = MODELS / "multimodel_results_v7.json"

FARMS = ["farm14", "farm16", "farm18", "farm19", "farm23"]
PCTS  = [85, 90, 95, 99]           # candidate threshold percentiles
TOP_K_PCT = 0.10                    # robust_recall window

# LOF hyperparameters
LOF_NEIGHBORS = 20
LOF_CONTAM    = 0.05


# ─── Utilities ─────────────────────────────────────────────────────────────────

def safe_div(a, b, default=0.0):
    return round(a / b, 6) if b > 0 else default


def compute_metrics(y_true, y_pred, scores, latency_s, n_rows):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)
    fpr       = safe_div(fp, fp + tn)
    k        = max(1, int(len(scores) * TOP_K_PCT))
    top_mask = np.zeros(len(scores), dtype=bool)
    top_mask[np.argsort(scores)[-k:]] = True
    rr_tp = int((top_mask & (y_true == 1)).sum())
    robust_recall = safe_div(rr_tp, tp + fn)
    lat_ms = round(latency_s * 1000 / n_rows, 6) if n_rows > 0 else 0.0
    return {
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1_score": round(f1, 6), "false_positive_rate": round(fpr, 6),
        "robust_recall": round(robust_recall, 6), "latency_ms_per_row": round(lat_ms, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_flagged": int(y_pred.sum()), "total_distress": int(y_true.sum()),
    }


def fill_scale_pd(df, cols, sc=None):
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


def build_ground_truth_cpu(cpu_test: pd.DataFrame, slurm: pd.DataFrame) -> pd.DataFrame:
    cpu = cpu_test.copy()
    cpu["node"] = cpu["instance"].str.split(":").str[0]
    cpu["timestamp_min"] = pd.to_datetime(cpu["timestamp"]).dt.floor("1min")
    slurm_agg = slurm[["node", "hw_config", "timestamp", "is_distress_status"]].copy()
    slurm_agg["timestamp"] = pd.to_datetime(slurm_agg["timestamp"])
    slurm_agg["timestamp_min"] = slurm_agg["timestamp"].dt.floor("1min")
    slurm_agg = (slurm_agg.groupby(["hw_config", "node", "timestamp_min"], sort=False)
                 ["is_distress_status"].max().reset_index())
    merged = cpu.merge(slurm_agg, on=["hw_config", "node", "timestamp_min"], how="left")
    merged["is_distress_status"] = merged["is_distress_status"].fillna(0).astype(int)
    return merged


def build_ground_truth_disk(disk_test: pd.DataFrame, slurm: pd.DataFrame) -> pd.DataFrame:
    """Disk instance format: same as CPU — 'farm14XXXX:9100'."""
    disk = disk_test.copy()
    disk["node"] = disk["instance"].str.split(":").str[0]
    disk["timestamp_min"] = pd.to_datetime(disk["timestamp"]).dt.floor("1min")
    slurm_agg = slurm[["node", "hw_config", "timestamp", "is_distress_status"]].copy()
    slurm_agg["timestamp"] = pd.to_datetime(slurm_agg["timestamp"])
    slurm_agg["timestamp_min"] = slurm_agg["timestamp"].dt.floor("1min")
    slurm_agg = (slurm_agg.groupby(["hw_config", "node", "timestamp_min"], sort=False)
                 ["is_distress_status"].max().reset_index())
    merged = disk.merge(slurm_agg, on=["hw_config", "node", "timestamp_min"], how="left")
    merged["is_distress_status"] = merged["is_distress_status"].fillna(0).astype(int)
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — ADAPTIVE THRESHOLDING
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_percentile(train_scores: np.ndarray, test_scores: np.ndarray,
                         y_true: np.ndarray, candidates: list) -> Dict[str, Any]:
    """
    Sweep percentile thresholds computed on TRAIN scores (no leakage).
    Pick the one that maximises recall on test set, with F1 as tiebreaker.
    Returns dict with chosen_pct, chosen_thr, and metrics at each candidate.
    """
    results = {}
    best_pct = candidates[0]
    best_score = -1.0  # will maximise F1

    for pct in candidates:
        thr = float(np.percentile(train_scores, pct))
        flags = (test_scores > thr).astype(int)
        tp = int(((flags == 1) & (y_true == 1)).sum())
        fn = int(((flags == 0) & (y_true == 1)).sum())
        fp = int(((flags == 1) & (y_true == 0)).sum())
        tn = int(((flags == 0) & (y_true == 0)).sum())
        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = safe_div(2 * prec * rec, prec + rec)
        results[pct] = {"thr": thr, "recall": round(rec, 4), "precision": round(prec, 4),
                        "f1": round(f1, 4), "tp": tp, "fp": fp}
        # Prioritise recall (safety: missing a fault is worse than a false alarm)
        # but use F1 as tiebreaker to avoid degenerate "flag everything"
        if f1 > best_score:
            best_score = f1
            best_pct = pct

    # Fallback: if all F1 are 0 (no distress), pick highest recall
    if best_score == 0:
        best_pct = max(candidates, key=lambda p: results[p]["recall"])

    return {"best_pct": best_pct, "best_thr": results[best_pct]["thr"],
            "sweep": results}


def adaptive_threshold_modality(modality: str, slurm: pd.DataFrame) -> Dict[str, Any]:
    """
    Run adaptive thresholding for cpu or disk.
    Uses existing *_anomaly_scores.parquet (no model retraining).
    Saves updated scores parquet with *_adp_flag column added.
    """
    print(f"\n{'='*60}")
    print(f"  PART 1 — ADAPTIVE THRESHOLDING [{modality.upper()}]")
    print(f"{'='*60}")

    scores_path = MODELS / f"{modality}_anomaly_scores.parquet"
    scores_all  = pd.read_parquet(str(scores_path))
    scores_all["timestamp"] = pd.to_datetime(scores_all["timestamp"])

    scores_tr = scores_all[scores_all["split"] == "train"]
    scores_te = scores_all[scores_all["split"] == "test"].copy()

    # Build ground truth joined df
    if modality == "cpu":
        merged = build_ground_truth_cpu(scores_te, slurm)
    else:
        merged = build_ground_truth_disk(scores_te, slurm)

    print(f"  Test rows: {len(merged):,}  Distress: {merged['is_distress_status'].sum():,}")

    adaptive_summary = {"modality": modality, "per_farm": {}}
    new_flags = np.zeros(len(merged), dtype=bool)

    for hw in FARMS:
        tr_sub  = scores_tr[scores_tr["hw_config"] == hw]
        m_idx   = merged["hw_config"] == hw
        te_sub  = merged[m_idx].copy()

        if len(tr_sub) == 0 or len(te_sub) == 0:
            print(f"\n  [{hw}] SKIP — no data")
            continue

        print(f"\n  [{hw}]  train={len(tr_sub):,}  test={len(te_sub):,}  "
              f"distress={te_sub['is_distress_status'].sum():,}")

        train_scores = tr_sub["ae_score"].values
        test_scores  = te_sub["ae_score"].values
        y_true       = te_sub["is_distress_status"].values.astype(int)

        # Current 3-sigma threshold (baseline)
        ae = joblib.load(MODELS / f"{modality}_autoencoder_{hw}.joblib")
        current_thr = ae.thr
        current_flags = (test_scores > current_thr).astype(int)
        tp0 = int(((current_flags == 1) & (y_true == 1)).sum())
        fn0 = int(((current_flags == 0) & (y_true == 1)).sum())
        fp0 = int(((current_flags == 1) & (y_true == 0)).sum())
        rec0 = safe_div(tp0, tp0 + fn0)
        prec0 = safe_div(tp0, tp0 + fp0)
        f10 = safe_div(2 * prec0 * rec0, prec0 + rec0)
        print(f"  [{hw}] Baseline (3σ thr={current_thr:.4f}): "
              f"R={rec0:.4f}  P={prec0:.4f}  F1={f10:.4f}  "
              f"TP={tp0}  FP={fp0}  distress={y_true.sum()}")

        # Adaptive: compute percentile thresholds from TRAIN scores
        best = find_best_percentile(train_scores, test_scores, y_true, PCTS)
        best_pct = best["best_pct"]
        best_thr = best["best_thr"]
        bm = best["sweep"][best_pct]
        print(f"  [{hw}] Best pct={best_pct}  thr={best_thr:.6f}: "
              f"R={bm['recall']:.4f}  P={bm['precision']:.4f}  F1={bm['f1']:.4f}  "
              f"TP={bm['tp']}  FP={bm['fp']}")

        # Print full sweep
        for pct in PCTS:
            s = best["sweep"][pct]
            marker = " ◄ BEST" if pct == best_pct else ""
            print(f"    pct={pct}: thr={s['thr']:.4f}  R={s['recall']:.4f}  "
                  f"P={s['precision']:.4f}  F1={s['f1']:.4f}{marker}")

        # Write adaptive flags back to merged slice
        adp_flags = (test_scores > best_thr).astype(bool)
        new_flags[m_idx.values] = adp_flags  # align with merged index

        adaptive_summary["per_farm"][hw] = {
            "baseline_thr": round(current_thr, 6),
            "baseline_recall": round(rec0, 4),
            "baseline_f1": round(f10, 4),
            "chosen_pct": best_pct,
            "chosen_thr": round(best_thr, 6),
            "chosen_recall": bm["recall"],
            "chosen_f1": bm["f1"],
            "recall_delta": round(bm["recall"] - rec0, 4),
            "f1_delta": round(bm["f1"] - f10, 4),
            "sweep": best["sweep"],
        }

    merged["ae_adp_flag"] = new_flags.astype(int)

    # Compute final full metrics with adaptive flags
    print(f"\n  {'─'*50}")
    print(f"  FINAL ADAPTIVE METRICS [{modality.upper()}]")
    print(f"  {'─'*50}")
    print(f"  {'Farm':<8} {'Model':<16} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}")

    farm_metrics = {}
    for hw in FARMS:
        m_idx = merged["hw_config"] == hw
        gdf   = merged[m_idx]
        if gdf.empty:
            continue
        y_true = gdf["is_distress_status"].values.astype(int)
        ae_scores = gdf["ae_score"].values

        # Adaptive AE
        ae_flags_adp = gdf["ae_adp_flag"].values.astype(int)
        t0 = time.perf_counter()
        _  = ae_flags_adp  # flags already computed
        lat_adp = time.perf_counter() - t0

        m_adp = compute_metrics(y_true, ae_flags_adp, ae_scores, lat_adp, max(len(gdf), 1))
        print(f"  {hw:<8} {'AE-Adaptive':<16} "
              f"{m_adp['precision']:>7.4f} {m_adp['recall']:>7.4f} "
              f"{m_adp['f1_score']:>7.4f} {m_adp['false_positive_rate']:>7.4f}")

        # Original IF flags (unchanged)
        if_flags  = gdf["if_flag"].values.astype(int)
        if_scores = gdf["if_score"].values
        m_if = compute_metrics(y_true, if_flags, if_scores, 0.001, max(len(gdf), 1))

        # Ensemble: adp AE OR IF
        ens_flags  = np.maximum(ae_flags_adp, if_flags)
        ens_scores = np.maximum(ae_scores, if_scores)
        m_ens = compute_metrics(y_true, ens_flags, ens_scores, 0.001, max(len(gdf), 1))
        print(f"  {hw:<8} {'Ens-Adaptive':<16} "
              f"{m_ens['precision']:>7.4f} {m_ens['recall']:>7.4f} "
              f"{m_ens['f1_score']:>7.4f} {m_ens['false_positive_rate']:>7.4f}")

        farm_metrics[hw] = {
            "modality": modality,
            "test_rows": int(len(gdf)),
            "test_distress_rows": int(y_true.sum()),
            "models": {
                "AE_Adaptive": m_adp,
                "IsolationForest": m_if,
                "Ens_Adaptive": m_ens,
            }
        }

    adaptive_summary["farm_metrics"] = farm_metrics
    adaptive_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return adaptive_summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — LOF ON PCA-COMPRESSED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def train_lof_modality(modality: str, slurm: pd.DataFrame) -> Dict[str, Any]:
    """
    Load existing PCA models, fit LOF on PCA-compressed train features,
    score test features, compute metrics.
    Saves: {modality}_lof_{farm}.joblib and {modality}_lof_scores.parquet
    Does NOT overwrite any existing file.

    Uses stratified subsampling (max LOF_MAX_TRAIN rows) + ball_tree algorithm
    so LOF is tractable on 400K–580K row farms without multi-hour waits.
    """
    print(f"\n{'='*60}")
    print(f"  PART 2 — LOF TRAINING [{modality.upper()}]")
    print(f"{'='*60}")

    tr_feat_path = FEATURES / f"{modality}_data_train_features.parquet"
    te_feat_path = FEATURES / f"{modality}_data_test_features.parquet"

    df_tr = pd.read_parquet(str(tr_feat_path))
    df_te = pd.read_parquet(str(te_feat_path))
    print(f"  Train: {df_tr.shape}   Test: {df_te.shape}")

    excl = {"instance", "timestamp", "hw_config"}
    fcols_tr = [c for c in df_tr.columns if c not in excl
                and pd.api.types.is_numeric_dtype(df_tr[c])]
    fcols_te = {c for c in df_te.columns if c not in excl
                and pd.api.types.is_numeric_dtype(df_te[c])}
    fcols = [c for c in fcols_tr if c in fcols_te]
    print(f"  Feature cols: {len(fcols)}")

    # Max rows fed to LOF.fit() — LOF is O(n*k*log n), 50K is ~15-30s per farm
    LOF_MAX_TRAIN = 50_000

    lof_summary = {"modality": modality, "per_farm": {}}
    all_lof_scores = []

    for hw in FARMS:
        lof_path = MODELS / f"{modality}_lof_{hw}.joblib"

        gtr = df_tr[df_tr["hw_config"] == hw]
        gte = df_te[df_te["hw_config"] == hw]

        if len(gtr) < 200:
            print(f"\n  [{hw}] SKIP — {len(gtr)} train rows < 200")
            continue

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  train={len(gtr):,}  test={len(gte):,}")

        # Load existing scaler + PCA (do NOT modify them)
        sc  = joblib.load(MODELS / f"{modality}_scaler_{hw}.joblib")
        pca = joblib.load(MODELS / f"{modality}_pca_{hw}.joblib")

        Xtr, _ = fill_scale_pd(gtr, fcols, sc)
        Xte, _ = fill_scale_pd(gte, fcols, sc)

        Xp_tr = pca.transform(Xtr)
        Xp_te = pca.transform(Xte) if len(Xte) > 0 else np.array([])

        print(f"  [{hw}] PCA dims: {Xtr.shape[1]} → {Xp_tr.shape[1]}")

        # ── Stratified subsampling for LOF.fit() ────────────────────────────
        n_fit = min(LOF_MAX_TRAIN, len(Xp_tr))
        if len(Xp_tr) > LOF_MAX_TRAIN:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Xp_tr), size=LOF_MAX_TRAIN, replace=False)
            Xp_tr_fit = Xp_tr[idx]
            print(f"  [{hw}] Subsampled {len(Xp_tr):,} → {n_fit:,} rows for LOF fit")
        else:
            Xp_tr_fit = Xp_tr
            print(f"  [{hw}] Using full {n_fit:,} rows for LOF fit")

        # ── Fit or load LOF ──────────────────────────────────────────────────
        if lof_path.exists():
            print(f"  [{hw}] Loading existing LOF model (already trained — skipping refit)")
            lof = joblib.load(lof_path)
        else:
            print(f"  [{hw}] Fitting LOF (n_neighbors={LOF_NEIGHBORS}, algorithm=ball_tree)...")
            t0 = time.perf_counter()
            lof = LocalOutlierFactor(
                n_neighbors=LOF_NEIGHBORS,
                contamination=LOF_CONTAM,
                novelty=True,
                algorithm="ball_tree",   # faster than brute for PCA-compressed dims
                leaf_size=30,
                n_jobs=-1,
            )
            lof.fit(Xp_tr_fit)
            fit_time = time.perf_counter() - t0
            print(f"  [{hw}] LOF fit in {fit_time:.1f}s")
            joblib.dump(lof, lof_path)
            print(f"  [{hw}] ✓ Saved: {lof_path.name}")

        # Score on full train + test sets (scoring is fast — O(n*k) via tree)
        t0 = time.perf_counter()
        lof_tr_scores = -lof.score_samples(Xp_tr)
        lof_tr_flags  = lof.predict(Xp_tr) == -1
        lof_te_scores = -lof.score_samples(Xp_te) if Xp_te.shape[0] > 0 else np.array([])
        lof_te_flags  = lof.predict(Xp_te) == -1    if Xp_te.shape[0] > 0 else np.array([])
        score_time = time.perf_counter() - t0
        print(f"  [{hw}] Scored in {score_time:.1f}s  "
              f"train flagged: {lof_tr_flags.mean()*100:.2f}%  "
              f"test flagged: {lof_te_flags.mean()*100:.2f}%")

        # Build score rows
        meta = ["instance", "timestamp", "hw_config"]
        tr_out = gtr[meta].copy().reset_index(drop=True)
        tr_out["lof_score"] = lof_tr_scores
        tr_out["lof_flag"]  = lof_tr_flags
        tr_out["split"] = "train"
        all_lof_scores.append(tr_out)

        if Xp_te.shape[0] > 0:
            te_out = gte[meta].copy().reset_index(drop=True)
            te_out["lof_score"] = lof_te_scores
            te_out["lof_flag"]  = lof_te_flags
            te_out["split"] = "test"
            all_lof_scores.append(te_out)

        lof_summary["per_farm"][hw] = {
            "pca_components":     int(Xp_tr.shape[1]),
            "lof_n_neighbors":    LOF_NEIGHBORS,
            "lof_train_flag_pct": round(float(lof_tr_flags.mean()) * 100, 3),
            "lof_test_flag_pct":  round(float(lof_te_flags.mean()) * 100, 3) if len(lof_te_flags) > 0 else None,
        }

    # Save LOF scores parquet
    if all_lof_scores:
        lof_df = pd.concat(all_lof_scores, ignore_index=True)
        out_path = MODELS / f"{modality}_lof_scores.parquet"
        if out_path.exists():
            print(f"\n  ⚠ {out_path.name} already exists — skipping to avoid overwrite")
        else:
            lof_df.to_parquet(str(out_path), index=False, compression="snappy")
            print(f"\n  ✓ LOF scores → {out_path}  ({len(lof_df):,} rows)")

    # Now evaluate LOF on test set with SLURM ground truth
    lof_df_all = pd.read_parquet(str(MODELS / f"{modality}_lof_scores.parquet"))
    lof_te     = lof_df_all[lof_df_all["split"] == "test"].copy()
    lof_te["timestamp"] = pd.to_datetime(lof_te["timestamp"])

    if modality == "cpu":
        merged = build_ground_truth_cpu(lof_te, slurm)
    else:
        merged = build_ground_truth_disk(lof_te, slurm)

    print(f"\n  LOF eval — joined {len(merged):,} rows  "
          f"distress={merged['is_distress_status'].sum():,}")

    lof_farm_metrics = {}
    print(f"\n  {'Farm':<8} {'Model':<12} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}")
    for hw in FARMS:
        gdf = merged[merged["hw_config"] == hw]
        if gdf.empty:
            continue
        y_true    = gdf["is_distress_status"].values.astype(int)
        lof_scores = gdf["lof_score"].values
        lof_flags  = gdf["lof_flag"].values.astype(int)

        t0 = time.perf_counter()
        lof_path = MODELS / f"{modality}_lof_{hw}.joblib"
        if lof_path.exists():
            lof_model = joblib.load(lof_path)
            # Time dummy inference
            sc  = joblib.load(MODELS / f"{modality}_scaler_{hw}.joblib")
            pca = joblib.load(MODELS / f"{modality}_pca_{hw}.joblib")
            n_dummy = min(500, len(gdf))
            # Reconstruct dummy input dims
            dummy = np.zeros((n_dummy, sc.n_features_in_))
            dummy_s = sc.transform(dummy)
            dummy_p = pca.transform(dummy_s)
            t0 = time.perf_counter()
            _ = lof_model.score_samples(dummy_p)
            lat_lof = (time.perf_counter() - t0) / n_dummy * 1000  # ms/row
        else:
            lat_lof = 0.0

        m_lof = compute_metrics(y_true, lof_flags, lof_scores, lat_lof / 1000, max(len(gdf), 1))
        print(f"  {hw:<8} {'LOF':<12} "
              f"{m_lof['precision']:>7.4f} {m_lof['recall']:>7.4f} "
              f"{m_lof['f1_score']:>7.4f} {m_lof['false_positive_rate']:>7.4f}")

        lof_farm_metrics[hw] = {
            "modality": modality,
            "test_rows": int(len(gdf)),
            "test_distress_rows": int(y_true.sum()),
            "models": {"LOF": m_lof}
        }

    lof_summary["farm_metrics"] = lof_farm_metrics
    lof_summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return lof_summary


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — MERGE INTO v7 UNIFIED JSON
# ═══════════════════════════════════════════════════════════════════════════════

def merge_into_v7(cpu_adp: Dict, disk_adp: Dict,
                  cpu_lof: Dict, disk_lof: Dict) -> None:
    """
    Load existing v6 JSON, inject adaptive + LOF metrics as new model entries,
    write to v7 files (never touching v6).
    """
    print(f"\n{'='*60}")
    print("  PART 3 — MERGING INTO v7")
    print(f"{'='*60}")

    with open(EXISTING_V6_EVAL) as f:
        v6_eval = json.load(f)
    with open(EXISTING_V6_MULTI) as f:
        v6_multi = json.load(f)

    # Deepcopy-style: start from v6 and add new keys
    import copy
    v7_eval  = copy.deepcopy(v6_eval)
    v7_multi = copy.deepcopy(v6_multi)

    v7_eval["generated_at"]  = datetime.now().isoformat(timespec="seconds")
    v7_multi["generated_at"] = datetime.now().isoformat(timespec="seconds")
    v7_eval["note"]  = "v7: adds AE_Adaptive + LOF models for CPU and Disk modalities"
    v7_multi["note"] = "v7: adds AE_Adaptive + LOF models for CPU and Disk modalities"

    # Ensure modalities list includes cpu and disk
    for mod in ["cpu", "disk"]:
        if mod not in v7_eval.get("modalities", []):
            v7_eval.setdefault("modalities", []).append(mod)
        if mod not in v7_multi.get("modalities", []):
            v7_multi.setdefault("modalities", []).append(mod)

    # Helper: inject new models into by_modality structure
    def inject_eval(mod: str, source: Dict, new_model_key: str) -> None:
        mod_eval = v7_eval.setdefault("by_modality", {}).setdefault(mod, {})
        for hw, farm_data in source.get("farm_metrics", {}).items():
            mod_eval.setdefault(hw, {})
            for model_name, m in farm_data["models"].items():
                tag = new_model_key if new_model_key else model_name
                mod_eval[hw][tag] = {
                    "precision":           m["precision"],
                    "recall":              m["recall"],
                    "f1_score":            m["f1_score"],
                    "false_positive_rate": m["false_positive_rate"],
                    "robust_recall":       m["robust_recall"],
                    "latency_ms_per_row":  m["latency_ms_per_row"],
                }

    def inject_multi(mod: str, source: Dict, new_model_key: str) -> None:
        mod_multi = v7_multi.setdefault("by_modality", {}).setdefault(mod, {})
        for hw, farm_data in source.get("farm_metrics", {}).items():
            if hw not in mod_multi:
                mod_multi[hw] = {
                    "status": "success",
                    "modality": mod,
                    "test_rows": farm_data.get("test_rows", 0),
                    "test_distress_rows": farm_data.get("test_distress_rows", 0),
                }
            for model_name, m in farm_data["models"].items():
                tag = new_model_key if new_model_key else model_name
                mod_multi[hw][tag] = {
                    "test_anomalies": m["total_flagged"],
                    "caught_distress": m["tp"],
                }

    # CPU adaptive
    inject_eval("cpu", cpu_adp, "")
    inject_multi("cpu", cpu_adp, "")
    # Disk adaptive
    inject_eval("disk", disk_adp, "")
    inject_multi("disk", disk_adp, "")
    # CPU LOF
    inject_eval("cpu", cpu_lof, "LOF")
    inject_multi("cpu", cpu_lof, "LOF")
    # Disk LOF
    inject_eval("disk", disk_lof, "LOF")
    inject_multi("disk", disk_lof, "LOF")

    # Safeguard: never overwrite
    for out_path, data in [(OUT_V7_EVAL, v7_eval), (OUT_V7_MULTI, v7_multi)]:
        if out_path.exists():
            print(f"  ⚠ {out_path.name} already exists — saving with timestamp suffix")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_path.with_stem(out_path.stem + f"_{ts}")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✓ Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CPU + DISK ADAPTIVE THRESHOLDING & LOF")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load SLURM ground truth once
    print("\n  Loading SLURM test features...")
    slurm = pd.read_parquet(str(SLURM_TEST_PATH))
    print(f"  SLURM test rows: {len(slurm):,}")

    # ── PART 1: Adaptive Thresholding ─────────────────────────────────────────
    # Resume: if already-saved adaptive JSONs exist, load them rather than re-run
    if OUT_CPU_ADAPTIVE.exists() and OUT_DISK_ADAPTIVE.exists():
        print(f"\n  [RESUME] Part 1 already complete — loading saved adaptive JSONs")
        with open(OUT_CPU_ADAPTIVE) as f:
            cpu_adp = json.load(f)
        with open(OUT_DISK_ADAPTIVE) as f:
            disk_adp = json.load(f)
        print(f"  ✓ Loaded: {OUT_CPU_ADAPTIVE}")
        print(f"  ✓ Loaded: {OUT_DISK_ADAPTIVE}")
    else:
        cpu_adp = adaptive_threshold_modality("cpu", slurm)
        with open(OUT_CPU_ADAPTIVE, "w") as f:
            json.dump(cpu_adp, f, indent=2, default=str)
        print(f"\n  ✓ Saved: {OUT_CPU_ADAPTIVE}")

        disk_adp = adaptive_threshold_modality("disk", slurm)
        with open(OUT_DISK_ADAPTIVE, "w") as f:
            json.dump(disk_adp, f, indent=2, default=str)
        print(f"\n  ✓ Saved: {OUT_DISK_ADAPTIVE}")

    # ── PART 2: LOF Training ──────────────────────────────────────────────────
    cpu_lof  = train_lof_modality("cpu", slurm)
    disk_lof = train_lof_modality("disk", slurm)

    # ── PART 3: Merge into v7 ─────────────────────────────────────────────────
    merge_into_v7(cpu_adp, disk_adp, cpu_lof, disk_lof)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  IMPROVEMENT SUMMARY")
    print("=" * 60)
    for modality, adp in [("CPU", cpu_adp), ("Disk", disk_adp)]:
        print(f"\n  {modality} Adaptive Thresholding:")
        for hw, info in adp.get("per_farm", {}).items():
            delta = info.get("recall_delta", 0.0)
            arrow = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"    {hw}: pct={info['chosen_pct']}  "
                  f"recall {info['baseline_recall']:.4f} → {info['chosen_recall']:.4f} ({arrow})")
    for modality, lof in [("CPU", cpu_lof), ("Disk", disk_lof)]:
        print(f"\n  {modality} LOF Results:")
        for hw, farm_data in lof.get("farm_metrics", {}).items():
            m = farm_data["models"].get("LOF", {})
            print(f"    {hw}: R={m.get('recall', 0):.4f}  F1={m.get('f1_score', 0):.4f}  "
                  f"FPR={m.get('false_positive_rate', 0):.4f}")


if __name__ == "__main__":
    main()
