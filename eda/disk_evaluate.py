"""
eda/disk_evaluate.py
====================
Compute full evaluation metrics for disk anomaly models, then merge into
unified evaluation_report_metrics_v5.json → v6.json and
multimodel_results_v5.json → v6.json.

Ground truth: is_distress_status from slurm_data_test_features.parquet
Join key   : hw_config + timestamp (floor to 1-min) + node prefix from instance
"""
from __future__ import annotations
import json, time, datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from models.autoencoder import ReconAE  # noqa — for disk autoencoder unpickling
from eda.cpu_pipeline_v3 import ReconAE  # noqa — fallback

MODELS_DIR   = Path("artifacts/models")
FEATURES_DIR = Path("artifacts/features")

DISK_SCORES  = MODELS_DIR / "disk_anomaly_scores.parquet"
SLURM_TEST   = FEATURES_DIR / "slurm_data_test_features.parquet"

EXISTING_EVAL  = MODELS_DIR / "evaluation_report_metrics_v5.json"
EXISTING_MULTI = MODELS_DIR / "multimodel_results_v5.json"
OUT_EVAL       = MODELS_DIR / "evaluation_report_metrics_v6.json"
OUT_MULTI      = MODELS_DIR / "multimodel_results_v6.json"
DISK_EVAL_ONLY = MODELS_DIR / "disk_evaluation_report.json"

TOP_K_PCT = 0.10


def safe_div(a, b, d=0.0):
    return round(a / b, 6) if b > 0 else d


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
    top_idx  = np.argsort(scores)[-k:]
    top_mask = np.zeros(len(scores), dtype=bool)
    top_mask[top_idx] = True
    rr_tp    = int((top_mask & (y_true == 1)).sum())
    robust_recall = safe_div(rr_tp, tp + fn)

    latency_ms = round(latency_s * 1000 / n_rows, 6) if n_rows > 0 else 0.0

    return {
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1_score": round(f1, 6), "false_positive_rate": round(fpr, 6),
        "robust_recall": round(robust_recall, 6),
        "latency_ms_per_row": round(latency_ms, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total_flagged": int(y_pred.sum()), "total_distress": int(y_true.sum()),
    }


def benchmark_inference(hw, model_type, X):
    model_path  = MODELS_DIR / f"disk_{model_type}_{hw}.joblib"
    scaler_path = MODELS_DIR / f"disk_scaler_{hw}.joblib"
    pca_path    = MODELS_DIR / f"disk_pca_{hw}.joblib"
    if not model_path.exists(): return 0.0
    model = joblib.load(model_path)
    t0 = time.perf_counter()
    if scaler_path.exists() and pca_path.exists():
        sc = joblib.load(scaler_path); pca = joblib.load(pca_path)
        Xs = sc.transform(X)
        if model_type == "isoforest":
            _ = model.predict(pca.transform(Xs))
        else:
            _ = model.score(X)
    t1 = time.perf_counter()
    return t1 - t0


def build_ground_truth(disk_test, slurm):
    disk = disk_test.copy()
    disk["node"]          = disk["instance"].str.split(":").str[0]
    disk["timestamp_min"] = pd.to_datetime(disk["timestamp"]).dt.floor("1min")

    slurm_agg = slurm[["node","hw_config","timestamp","is_distress_status"]].copy()
    slurm_agg["timestamp"] = pd.to_datetime(slurm_agg["timestamp"])
    slurm_agg["timestamp_min"] = slurm_agg["timestamp"].dt.floor("1min")
    slurm_agg = (slurm_agg
                 .groupby(["hw_config","node","timestamp_min"], sort=False)["is_distress_status"]
                 .max().reset_index())

    merged = disk.merge(slurm_agg, on=["hw_config","node","timestamp_min"], how="left")
    merged["is_distress_status"] = merged["is_distress_status"].fillna(0).astype(int)
    return merged


def evaluate_disk():
    print("\n" + "="*60)
    print("  DISK EVALUATION — loading data")
    print("="*60)

    disk_all  = pd.read_parquet(str(DISK_SCORES))
    slurm     = pd.read_parquet(str(SLURM_TEST))
    disk_test = disk_all[disk_all["split"] == "test"].copy()

    print(f"  Disk test rows : {len(disk_test):,}")
    print(f"  SLURM rows     : {len(slurm):,}")
    print("\n  Joining disk scores with SLURM distress ground truth...")
    merged = build_ground_truth(disk_test, slurm)

    matched = (merged["is_distress_status"] > 0).sum()
    print(f"  Joined rows    : {len(merged):,}")
    print(f"  Distress rows  : {matched:,}  ({100*matched/len(merged):.3f}%)")

    farms   = sorted(merged["hw_config"].unique())
    results = {}

    for hw in farms:
        gdf    = merged[merged["hw_config"] == hw].copy()
        y_true = gdf["is_distress_status"].values.astype(int)
        n      = len(gdf)

        print(f"\n  {'─'*50}")
        print(f"  [{hw}]  n={n:,}  distress={y_true.sum():,}  ({100*y_true.mean():.3f}%)")

        farm_result = {
            "modality": "disk", "test_rows": int(n),
            "test_distress_rows": int(y_true.sum()), "models": {}
        }

        # IsolationForest
        if_scores = gdf["if_score"].values
        if_flags  = gdf["if_flag"].values.astype(int)
        lat_if    = benchmark_inference(hw, "isoforest", np.zeros((min(500,n), 154)))
        m_if      = compute_metrics(y_true, if_flags, if_scores, lat_if, min(500,n))
        farm_result["models"]["IsolationForest"] = m_if
        print(f"  [{hw}] IF  — P={m_if['precision']:.4f}  R={m_if['recall']:.4f}  "
              f"F1={m_if['f1_score']:.4f}  FPR={m_if['false_positive_rate']:.4f}  "
              f"RR={m_if['robust_recall']:.4f}")

        # Autoencoder
        ae_scores = gdf["ae_score"].values
        ae_flags  = gdf["ae_flag"].values.astype(int)
        lat_ae    = benchmark_inference(hw, "autoencoder", np.zeros((min(500,n), 154)))
        m_ae      = compute_metrics(y_true, ae_flags, ae_scores, lat_ae, min(500,n))
        farm_result["models"]["Autoencoder"] = m_ae
        print(f"  [{hw}] AE  — P={m_ae['precision']:.4f}  R={m_ae['recall']:.4f}  "
              f"F1={m_ae['f1_score']:.4f}  FPR={m_ae['false_positive_rate']:.4f}  "
              f"RR={m_ae['robust_recall']:.4f}")

        # Ensemble
        ens_flags  = np.maximum(if_flags, ae_flags)
        ens_scores = np.maximum(if_scores, ae_scores / (ae_scores.max() + 1e-9))
        m_ens      = compute_metrics(y_true, ens_flags, ens_scores, lat_if+lat_ae, min(500,n))
        farm_result["models"]["Ensemble"] = m_ens
        print(f"  [{hw}] ENS — P={m_ens['precision']:.4f}  R={m_ens['recall']:.4f}  "
              f"F1={m_ens['f1_score']:.4f}  FPR={m_ens['false_positive_rate']:.4f}  "
              f"RR={m_ens['robust_recall']:.4f}")

        results[hw] = farm_result

    return results


def main():
    disk_results = evaluate_disk()

    # Flat format for v6
    disk_eval_fmt  = {hw: {m: {k: v for k, v in met.items()
                               if k in ["precision","recall","f1_score",
                                        "false_positive_rate","robust_recall",
                                        "latency_ms_per_row"]}
                           for m, met in farm["models"].items()}
                     for hw, farm in disk_results.items()}

    disk_multi_fmt = {hw: {"status": "success", "modality": "disk",
                            "test_rows": farm["test_rows"],
                            "test_distress_rows": farm["test_distress_rows"],
                            **{m: {"test_anomalies": met["total_flagged"],
                                   "caught_distress": met["tp"]}
                               for m, met in farm["models"].items()}}
                     for hw, farm in disk_results.items()}

    # Save disk-only
    disk_only = {
        "modality": "disk_data",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": "Ground truth: is_distress_status from SLURM test features",
        "per_farm": disk_results,
    }
    with open(DISK_EVAL_ONLY, "w") as f:
        json.dump(disk_only, f, indent=2, default=str)
    print(f"\n  Saved: {DISK_EVAL_ONLY}")

    # Merge into v6
    print("\n" + "="*60)
    print("  MERGING with memory+slurm+cpu metrics (v5 -> v6)")
    print("="*60)

    with open(EXISTING_EVAL) as f:  existing_eval  = json.load(f)
    with open(EXISTING_MULTI) as f: existing_multi = json.load(f)

    # Build v6 eval
    v6_eval = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "modalities": ["memory_slurm", "cpu", "disk"],
        "note": "memory_slurm from v4; cpu from cpu_evaluate.py; disk newly computed",
        "by_modality": {
            **existing_eval.get("by_modality", {}),
            "disk": disk_eval_fmt,
        },
        "combined_per_farm": {}
    }

    # Rebuild combined_per_farm
    combined = {}
    for mod, farm_dict in v6_eval["by_modality"].items():
        for hw, models in farm_dict.items():
            if hw not in combined: combined[hw] = {}
            for model_name, metrics in models.items():
                key = f"{model_name}_{mod}" if mod != "memory_slurm" else model_name
                combined[hw][key] = {**metrics, "modality": mod}
    v6_eval["combined_per_farm"] = combined

    with open(OUT_EVAL, "w") as f:
        json.dump(v6_eval, f, indent=2, default=str)
    print(f"  Saved: {OUT_EVAL}")

    # Build v6 multi
    v6_multi = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "modalities": ["memory_slurm", "cpu", "disk"],
        "by_modality": {
            **existing_multi.get("by_modality", {}),
            "disk": disk_multi_fmt,
        }
    }
    with open(OUT_MULTI, "w") as f:
        json.dump(v6_multi, f, indent=2, default=str)
    print(f"  Saved: {OUT_MULTI}")

    # Print summary table
    print("\n" + "="*60)
    print("  FINAL DISK METRICS SUMMARY")
    print("="*60)
    print(f"  {'Farm':<8} {'Model':<18} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7} {'RobR':>7} {'Lat(ms)':>9}")
    print(f"  {'-'*72}")
    for hw, models in disk_eval_fmt.items():
        for model_name, m in models.items():
            print(f"  {hw:<8} {model_name:<18} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
                  f"{m['f1_score']:>7.4f} {m['false_positive_rate']:>7.4f} "
                  f"{m['robust_recall']:>7.4f} {m['latency_ms_per_row']:>9.4f}")

    print(f"\n  Output files:")
    print(f"    {DISK_EVAL_ONLY}")
    print(f"    {OUT_EVAL}  (memory+slurm+cpu+disk combined)")
    print(f"    {OUT_MULTI}")


if __name__ == "__main__":
    main()
