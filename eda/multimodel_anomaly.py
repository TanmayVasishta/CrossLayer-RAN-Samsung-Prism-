"""
eda/multimodel_anomaly.py
=========================
Multimodel anomaly detection pipeline.
Trains and compares three separate unsupervised models per hardware configuration:
1. Isolation Forest (Tree-based)
2. Local Outlier Factor (Density-based)
3. Autoencoder (Neural Network via MLPRegressor)
"""
from __future__ import annotations

import argparse
import json
import time
import joblib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from eda.lib import ensure_dir

def safe_div(n: float, d: float) -> float:
    return float(n / d) if d > 0 else 0.0

def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, hw_config: str, out_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    meta_cols = ["node", "timestamp", "hw_config", "is_distress_status", "status", "job"]
    
    if len(train_df) < 10:
        return {"status": "skipped", "reason": "Not enough training data"}, {}

    feat_cols = [c for c in train_df.columns if c not in meta_cols]
    
    X_train = train_df[feat_cols]
    
    if not test_df.empty:
        for c in feat_cols:
            if c not in test_df.columns:
                test_df[c] = 0.0
        X_test = test_df[feat_cols]
    else:
        X_test = pd.DataFrame(columns=feat_cols)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if not X_test.empty else None

    # Save scaler just in case
    joblib.dump(scaler, out_dir / f"scaler_{hw_config}.joblib")

    results = {
        "status": "success",
        "train_rows": len(X_train),
        "test_rows": len(X_test) if X_test_scaled is not None else 0,
        "models": {}
    }
    
    metrics = {}
    
    distress_mask = test_df["is_distress_status"] == 1 if "is_distress_status" in test_df.columns else None
    total_distress = int(distress_mask.sum()) if distress_mask is not None else 0
    results["test_distress_rows"] = total_distress
    
    noise_std = 0.01
    X_test_noisy = None
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        X_test_noisy = X_test_scaled + np.random.normal(0, noise_std, X_test_scaled.shape)

    # 1. Isolation Forest
    print(f"    Training Isolation Forest...")
    iso = IsolationForest(n_estimators=100, max_samples="auto", contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_train_scaled)
    joblib.dump(iso, out_dir / f"isoforest_{hw_config}.joblib")
    
    results["models"]["IsolationForest"] = {}
    metrics["IsolationForest"] = {}
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        t0 = time.time()
        iso_preds = iso.predict(X_test_scaled)
        t1 = time.time()
        
        iso_anomalies = int((iso_preds == -1).sum())
        results["models"]["IsolationForest"]["test_anomalies"] = iso_anomalies
        
        iso_caught = int((iso_preds[distress_mask] == -1).sum()) if total_distress > 0 else 0
        results["models"]["IsolationForest"]["caught_distress"] = iso_caught
        
        # Metrics
        tp = iso_caught
        fp = iso_anomalies - iso_caught
        fn = total_distress - iso_caught
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        latency = ((t1 - t0) * 1000) / len(X_test_scaled)
        
        # Robustness
        iso_robust_recall = None
        if total_distress > 0:
            iso_preds_noisy = iso.predict(X_test_noisy)
            iso_robust_caught = int((iso_preds_noisy[distress_mask] == -1).sum())
            iso_robust_recall = safe_div(iso_robust_caught, total_distress)
            
        metrics["IsolationForest"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "latency_ms_per_row": latency,
            "robust_recall": iso_robust_recall
        }

    # 2. Local Outlier Factor (Uncompromised)
    print(f"    Training LOF (O(N^2) complexity)...")
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.01, n_jobs=-1)
    lof.fit(X_train_scaled)
    joblib.dump(lof, out_dir / f"lof_{hw_config}.joblib")
    
    results["models"]["LOF"] = {}
    metrics["LOF"] = {}
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        t0 = time.time()
        lof_preds = lof.predict(X_test_scaled)
        t1 = time.time()
        
        lof_anomalies = int((lof_preds == -1).sum())
        results["models"]["LOF"]["test_anomalies"] = lof_anomalies
        
        lof_caught = int((lof_preds[distress_mask] == -1).sum()) if total_distress > 0 else 0
        results["models"]["LOF"]["caught_distress"] = lof_caught
        
        # Metrics
        tp = lof_caught
        fp = lof_anomalies - lof_caught
        fn = total_distress - lof_caught
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        latency = ((t1 - t0) * 1000) / len(X_test_scaled)
        
        # Robustness
        lof_robust_recall = None
        if total_distress > 0:
            lof_preds_noisy = lof.predict(X_test_noisy)
            lof_robust_caught = int((lof_preds_noisy[distress_mask] == -1).sum())
            lof_robust_recall = safe_div(lof_robust_caught, total_distress)
            
        metrics["LOF"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "latency_ms_per_row": latency,
            "robust_recall": lof_robust_recall
        }

    # 3. Autoencoder
    print(f"    Training Autoencoder...")
    n_features = X_train_scaled.shape[1]
    hidden_layer_sizes = (max(2, n_features // 2),)
    ae = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=100,
        random_state=42,
        early_stopping=True
    )
    
    # Conditional Denoising for farm14
    if hw_config == "farm14":
        print(f"      -> Injecting Gaussian noise for farm14 Denoising Autoencoder...")
        X_train_noisy = X_train_scaled + np.random.normal(0, noise_std, X_train_scaled.shape)
        ae.fit(X_train_noisy, X_train_scaled)
    else:
        ae.fit(X_train_scaled, X_train_scaled)
        
    joblib.dump(ae, out_dir / f"autoencoder_{hw_config}.joblib")
    
    train_preds = ae.predict(X_train_scaled)
    train_err = np.mean(np.square(X_train_scaled - train_preds), axis=1)
    
    cluster_thresholds = {
        "farm14": 99,
        "farm16": 99,
        "farm18": 99,
        "farm19": 80,
        "farm23": 99
    }
    pct = cluster_thresholds.get(hw_config, 99)
    threshold = np.percentile(train_err, pct)
    
    results["models"]["Autoencoder"] = {}
    metrics["Autoencoder"] = {}
    if X_test_scaled is not None and len(X_test_scaled) > 0:
        t0 = time.time()
        test_preds = ae.predict(X_test_scaled)
        test_err = np.mean(np.square(X_test_scaled - test_preds), axis=1)
        t1 = time.time()
        
        ae_anomalies = int((test_err > threshold).sum())
        results["models"]["Autoencoder"]["test_anomalies"] = ae_anomalies
        
        ae_caught = int((test_err[distress_mask] > threshold).sum()) if total_distress > 0 else 0
        results["models"]["Autoencoder"]["caught_distress"] = ae_caught
        
        # Metrics
        tp = ae_caught
        fp = ae_anomalies - ae_caught
        fn = total_distress - ae_caught
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        latency = ((t1 - t0) * 1000) / len(X_test_scaled)
        
        # Robustness
        ae_robust_recall = None
        if total_distress > 0:
            test_preds_noisy = ae.predict(X_test_noisy)
            test_err_noisy = np.mean(np.square(X_test_noisy - test_preds_noisy), axis=1)
            ae_robust_caught = int((test_err_noisy[distress_mask] > threshold).sum())
            ae_robust_recall = safe_div(ae_robust_caught, total_distress)
            
        metrics["Autoencoder"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "latency_ms_per_row": latency,
            "robust_recall": ae_robust_recall
        }

    return results, metrics

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/models"))
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    train_path = args.features_dir / "joined_train_features.parquet"
    test_path = args.features_dir / "joined_test_features.parquet"

    if not train_path.exists():
        print(f"Error: {train_path} not found.")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()
    hw_configs = train_df["hw_config"].unique()
    
    summary_path = args.out_dir / "multimodel_results_v2.json"
    metrics_path = args.out_dir / "evaluation_report_metrics_v2.json"
    
    all_results = {}
    all_metrics = {}
    
    for hw in hw_configs:
        print(f"\n=== Multimodel Training for {hw} ===")
        hw_train = train_df[train_df["hw_config"] == hw].copy()
        hw_test = test_df[test_df["hw_config"] == hw].copy() if not test_df.empty else pd.DataFrame()
        
        res, met = train_and_evaluate(hw_train, hw_test, hw, args.out_dir)
        all_results[hw] = res
        all_metrics[hw] = met
        
        # Save incrementally
        with summary_path.open("w") as f:
            json.dump(all_results, f, indent=2)
        with metrics_path.open("w") as f:
            json.dump(all_metrics, f, indent=2)
            
        if res["status"] == "success":
            print(f"  Train: {res['train_rows']} rows | Test: {res['test_rows']} rows")
            for model_name, model_stats in res["models"].items():
                anomalies = model_stats.get("test_anomalies", 0)
                caught = model_stats.get("caught_distress", 0)
                print(f"  -> {model_name} flagged {anomalies} anomalies (Caught {caught} distress)")

    print(f"\nAll clusters complete. Models saved as .joblib in {args.out_dir}")
    print(f"Results saved to {summary_path}")
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
