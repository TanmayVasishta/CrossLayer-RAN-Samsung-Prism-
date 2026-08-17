"""
eda/cpu_evaluate.py
===================
Compute full evaluation metrics for CPU anomaly models, then merge with
the existing memory+slurm evaluation_report_metrics_v4.json and
multimodel_results_v4.json into unified v5 files.

Metrics computed per farm x per model (IsolationForest, Autoencoder, Ensemble):
  - precision
  - recall
  - f1_score
  - false_positive_rate (FPR)
  - robust_recall  (recall on top-10% scoring rows)
  - latency_ms_per_row

Ground truth: is_distress_status from slurm_data_test_features.parquet
Join key   : hw_config + timestamp (floor to 1-min) + node prefix from instance
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from eda.cpu_pipeline_v3 import ReconAE  # needed for joblib unpickling of autoencoder

# ─── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR   = Path("artifacts/models")
FEATURES_DIR = Path("artifacts/features")

CPU_SCORES   = MODELS_DIR / "cpu_anomaly_scores.parquet"
SLURM_TEST   = FEATURES_DIR / "slurm_data_test_features.parquet"

EXISTING_EVAL    = MODELS_DIR / "evaluation_report_metrics_v4.json"
EXISTING_MULTI   = MODELS_DIR / "multimodel_results_v4.json"

OUT_EVAL         = MODELS_DIR / "evaluation_report_metrics_v5.json"
OUT_MULTI        = MODELS_DIR / "multimodel_results_v5.json"
CPU_EVAL_ONLY    = MODELS_DIR / "cpu_evaluation_report.json"

TOP_K_PCT        = 0.10   # robust_recall: top 10% scores


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return round(a / b, 6) if b > 0 else default


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    scores: np.ndarray, latency_s: float, n_rows: int) -> Dict[str, Any]:
    """Compute all metrics given binary ground truth, predictions and raw scores."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)
    fpr       = safe_div(fp, fp + tn)

    # Robust recall: among top-K% by score, what fraction of true positives caught?
    k = max(1, int(len(scores) * TOP_K_PCT))
    top_idx  = np.argsort(scores)[-k:]
    top_mask = np.zeros(len(scores), dtype=bool)
    top_mask[top_idx] = True
    rr_tp    = int(((top_mask) & (y_true == 1)).sum())
    robust_recall = safe_div(rr_tp, tp + fn)

    latency_ms = round(latency_s * 1000 / n_rows, 6) if n_rows > 0 else 0.0

    return {
        "precision":          round(precision, 6),
        "recall":             round(recall, 6),
        "f1_score":           round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "robust_recall":      round(robust_recall, 6),
        "latency_ms_per_row": round(latency_ms, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_flagged":      int(y_pred.sum()),
        "total_distress":     int(y_true.sum()),
    }


def benchmark_inference(hw: str, model_type: str, X: np.ndarray) -> float:
    """Time inference for latency_ms_per_row."""
    model_path = MODELS_DIR / f"cpu_{model_type.lower()}_{hw}.joblib"
    if not model_path.exists():
        return 0.0
    model = joblib.load(model_path)

    # Load scaler + pca for proper preprocessing
    scaler_path = MODELS_DIR / f"cpu_scaler_{hw}.joblib"
    pca_path    = MODELS_DIR / f"cpu_pca_{hw}.joblib"

    t0 = time.perf_counter()
    if scaler_path.exists() and pca_path.exists():
        sc  = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        Xs  = sc.transform(X)
        if model_type == "isoforest":
            Xp = pca.transform(Xs)
            _ = model.predict(Xp)          # IsolationForest: predict()
        else:  # Autoencoder (ReconAE)
            _ = model.score(X)             # ReconAE: custom score()
    else:
        if hasattr(model, "predict"):
            _ = model.predict(X)
        elif hasattr(model, "score"):
            _ = model.score(X)
    t1 = time.perf_counter()
    return t1 - t0


def build_ground_truth(cpu_test: pd.DataFrame, slurm: pd.DataFrame) -> pd.DataFrame:
    """
    Join CPU test scores with SLURM distress ground truth.

    CPU:   instance='farm140105:9100'  hw_config='farm14'  timestamp=1-min
    SLURM: node='farm140105'           hw_config='farm14'  timestamp=sub-minute

    Strategy:
      1. Extract node prefix from CPU instance (strip port)
      2. Floor SLURM timestamps to 1-min
      3. Aggregate is_distress_status per (hw_config, node, timestamp_min) → max
      4. Join on (hw_config, node, timestamp_min)
    """
    cpu = cpu_test.copy()
    cpu["node"]          = cpu["instance"].str.split(":").str[0]
    cpu["timestamp_min"] = cpu["timestamp"].dt.floor("1min")

    slurm_agg = (slurm[["node", "hw_config", "timestamp", "is_distress_status"]]
                 .copy())
    slurm_agg["timestamp"] = pd.to_datetime(slurm_agg["timestamp"])
    slurm_agg["timestamp_min"] = slurm_agg["timestamp"].dt.floor("1min")
    slurm_agg = (slurm_agg
                 .groupby(["hw_config", "node", "timestamp_min"], sort=False)["is_distress_status"]
                 .max()
                 .reset_index())

    merged = cpu.merge(slurm_agg,
                       on=["hw_config", "node", "timestamp_min"],
                       how="left")
    merged["is_distress_status"] = merged["is_distress_status"].fillna(0).astype(int)
    return merged


def evaluate_cpu() -> Dict[str, Any]:
    """Main evaluation function. Returns per-farm metrics dict."""
    print("\n" + "="*60)
    print("  CPU EVALUATION — loading data")
    print("="*60)

    cpu_all  = pd.read_parquet(str(CPU_SCORES))
    slurm    = pd.read_parquet(str(SLURM_TEST))
    cpu_test = cpu_all[cpu_all["split"] == "test"].copy()
    cpu_test["timestamp"] = pd.to_datetime(cpu_test["timestamp"])

    print(f"  CPU test rows : {len(cpu_test):,}")
    print(f"  SLURM rows    : {len(slurm):,}")

    print("\n  Joining CPU scores with SLURM distress ground truth...")
    merged = build_ground_truth(cpu_test, slurm)
    matched = (merged["is_distress_status"] > 0).sum()
    total   = len(merged)
    print(f"  Joined rows   : {total:,}")
    print(f"  Distress rows : {matched:,}  ({100*matched/total:.3f}%)")

    farms   = sorted(merged["hw_config"].unique())
    results = {}

    for hw in farms:
        gdf = merged[merged["hw_config"] == hw].copy()
        y_true = gdf["is_distress_status"].values.astype(int)
        n      = len(gdf)

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  n={n:,}  distress={y_true.sum():,}  ({100*y_true.mean():.3f}%)")

        farm_result: Dict[str, Any] = {
            "modality":          "cpu",
            "test_rows":         int(n),
            "test_distress_rows": int(y_true.sum()),
            "models": {}
        }

        # ── IsolationForest ──────────────────────────────────────────────────
        if_scores = gdf["if_score"].values
        if_flags  = gdf["if_flag"].values.astype(int)

        lat_if = benchmark_inference(hw, "isoforest",
                                     np.zeros((min(1000, n), 182)))  # dummy for timing
        m_if   = compute_metrics(y_true, if_flags, if_scores, lat_if, min(1000, n))
        farm_result["models"]["IsolationForest"] = m_if
        print(f"  [{hw}] IF  — P={m_if['precision']:.4f}  R={m_if['recall']:.4f}  "
              f"F1={m_if['f1_score']:.4f}  FPR={m_if['false_positive_rate']:.4f}  "
              f"RR={m_if['robust_recall']:.4f}")

        # ── Autoencoder ──────────────────────────────────────────────────────
        ae_scores = gdf["ae_score"].values
        ae_flags  = gdf["ae_flag"].values.astype(int)

        lat_ae = benchmark_inference(hw, "autoencoder",
                                     np.zeros((min(1000, n), 182)))
        m_ae   = compute_metrics(y_true, ae_flags, ae_scores, lat_ae, min(1000, n))
        farm_result["models"]["Autoencoder"] = m_ae
        print(f"  [{hw}] AE  — P={m_ae['precision']:.4f}  R={m_ae['recall']:.4f}  "
              f"F1={m_ae['f1_score']:.4f}  FPR={m_ae['false_positive_rate']:.4f}  "
              f"RR={m_ae['robust_recall']:.4f}")

        # ── Ensemble (IF OR AE) ───────────────────────────────────────────────
        ens_flags  = np.maximum(if_flags, ae_flags)
        ens_scores = np.maximum(if_scores, ae_scores / (ae_scores.max() + 1e-9))
        m_ens      = compute_metrics(y_true, ens_flags, ens_scores, lat_if + lat_ae, min(1000, n))
        farm_result["models"]["Ensemble"] = m_ens
        print(f"  [{hw}] ENS — P={m_ens['precision']:.4f}  R={m_ens['recall']:.4f}  "
              f"F1={m_ens['f1_score']:.4f}  FPR={m_ens['false_positive_rate']:.4f}  "
              f"RR={m_ens['robust_recall']:.4f}")

        results[hw] = farm_result

    return results


def build_eval_report_format(cpu_results: Dict) -> Dict:
    """Convert to the same flat format as evaluation_report_metrics_v4.json."""
    out = {}
    for hw, farm in cpu_results.items():
        out[hw] = {}
        for model_name, m in farm["models"].items():
            out[hw][model_name] = {
                "precision":           m["precision"],
                "recall":              m["recall"],
                "f1_score":            m["f1_score"],
                "false_positive_rate": m["false_positive_rate"],
                "robust_recall":       m["robust_recall"],
                "latency_ms_per_row":  m["latency_ms_per_row"],
            }
    return out


def build_multi_format(cpu_results: Dict) -> Dict:
    """Convert to the same format as multimodel_results_v4.json."""
    out = {}
    for hw, farm in cpu_results.items():
        out[hw] = {
            "status":            "success",
            "modality":          "cpu",
            "test_rows":         farm["test_rows"],
            "test_distress_rows": farm["test_distress_rows"],
            "models": {}
        }
        for model_name, m in farm["models"].items():
            out[hw][model_name] = {
                "test_anomalies":  m["total_flagged"],
                "caught_distress": m["tp"],
            }
    return out


def main():
    import datetime

    # Run CPU evaluation
    cpu_results = evaluate_cpu()

    cpu_eval_fmt  = build_eval_report_format(cpu_results)
    cpu_multi_fmt = build_multi_format(cpu_results)

    # ── Save CPU-only evaluation ──────────────────────────────────────────────
    cpu_only = {
        "modality":     "cpu_data",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note":         "Ground truth: is_distress_status from SLURM test features",
        "per_farm":     cpu_results,
    }
    with open(CPU_EVAL_ONLY, "w") as f:
        json.dump(cpu_only, f, indent=2, default=str)
    print(f"\n  Saved: {CPU_EVAL_ONLY}")

    # ── Merge with existing memory+slurm v4 files ─────────────────────────────
    print("\n" + "="*60)
    print("  MERGING with existing memory+slurm metrics")
    print("="*60)

    # Load existing v4 eval report (memory + slurm per farm)
    with open(EXISTING_EVAL) as f:
        existing_eval = json.load(f)

    # Add cpu entries with modality tag
    merged_eval = {}
    # Memory+slurm farms tagged
    for hw, models in existing_eval.items():
        merged_eval[f"{hw}_memory_slurm"] = models

    # CPU farms
    for hw, models in cpu_eval_fmt.items():
        merged_eval[f"{hw}_cpu"] = models

    # Also keep flat merged (overwrite if same farm exists — CPU adds extra keys)
    flat_merged = {}
    for hw, models in existing_eval.items():
        flat_merged[hw] = {}
        for model_name, metrics in models.items():
            flat_merged[hw][model_name] = dict(metrics)
            flat_merged[hw][model_name]["modality"] = "memory_slurm"

    for hw, models in cpu_eval_fmt.items():
        if hw not in flat_merged:
            flat_merged[hw] = {}
        for model_name, metrics in models.items():
            cpu_key = f"{model_name}_cpu"
            flat_merged[hw][cpu_key] = dict(metrics)
            flat_merged[hw][cpu_key]["modality"] = "cpu"

    v5_eval = {
        "generated_at":   datetime.datetime.now().isoformat(timespec="seconds"),
        "modalities":     ["memory_slurm", "cpu"],
        "note":           "memory_slurm metrics from v4; cpu metrics newly computed using SLURM distress ground truth",
        "by_modality": {
            "memory_slurm": existing_eval,
            "cpu":          cpu_eval_fmt,
        },
        "combined_per_farm": flat_merged,
    }

    with open(OUT_EVAL, "w") as f:
        json.dump(v5_eval, f, indent=2, default=str)
    print(f"  Saved: {OUT_EVAL}")

    # Load existing v4 multimodel
    with open(EXISTING_MULTI) as f:
        existing_multi = json.load(f)

    v5_multi = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "modalities":   ["memory_slurm", "cpu"],
        "by_modality": {
            "memory_slurm": existing_multi,
            "cpu":          cpu_multi_fmt,
        }
    }

    with open(OUT_MULTI, "w") as f:
        json.dump(v5_multi, f, indent=2, default=str)
    print(f"  Saved: {OUT_MULTI}")

    # ── Print final summary table ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL CPU METRICS SUMMARY")
    print("="*60)
    print(f"  {'Farm':<8} {'Model':<18} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7} {'RobR':>7} {'Lat(ms)':>9}")
    print(f"  {'-'*72}")
    for hw, models in cpu_eval_fmt.items():
        for model_name, m in models.items():
            print(f"  {hw:<8} {model_name:<18} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1_score']:>7.4f} {m['false_positive_rate']:>7.4f} "
                  f"{m['robust_recall']:>7.4f} {m['latency_ms_per_row']:>9.4f}")

    print(f"\n  Output files:")
    print(f"    {CPU_EVAL_ONLY}")
    print(f"    {OUT_EVAL}  (memory+slurm+cpu combined)")
    print(f"    {OUT_MULTI} (multimodel counts combined)")


if __name__ == "__main__":
    main()
