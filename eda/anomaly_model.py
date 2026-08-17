"""
eda/anomaly_model.py
====================
Baseline unsupervised anomaly detection model (Isolation Forest).
Crucially, trains separate models per hardware configuration (hw_config)
to prevent flagging normal behaviour on lower-spec nodes as anomalous.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from eda.lib import ensure_dir

def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, hw_config: str) -> Dict[str, Any]:
    """
    Train an Isolation Forest baseline for a specific hardware configuration.
    """
    # Exclude metadata columns from modeling
    meta_cols = ["node", "timestamp", "hw_config", "is_distress_status", "status", "job"]
    
    # We might have missing test/train for a specific hw_config in small samples
    if len(train_df) == 0:
        return {"status": "skipped", "reason": "No training data"}

    feat_cols = [c for c in train_df.columns if c not in meta_cols]
    
    X_train = train_df[feat_cols]
    
    if not test_df.empty:
        for c in feat_cols:
            if c not in test_df.columns:
                test_df[c] = 0.0
        X_test = test_df[feat_cols]
    else:
        X_test = pd.DataFrame(columns=feat_cols)

    # Standardize based on normal train data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if not X_test.empty:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None

    # Isolation Forest setup
    # contamination is an estimate of the proportion of outliers in the data set
    # Since train data is strictly normal, we set it very low (e.g., 0.001)
    iso = IsolationForest(
        n_estimators=100, 
        max_samples="auto", 
        contamination=0.001, 
        random_state=42, 
        n_jobs=-1
    )
    
    iso.fit(X_train_scaled)
    
    # Predict (-1 is anomaly, 1 is normal)
    # We can also get anomaly scores (lower is more anomalous)
    train_scores = iso.score_samples(X_train_scaled)
    
    results = {
        "status": "success",
        "train_rows": len(X_train),
        "test_rows": len(X_test) if X_test_scaled is not None else 0,
        "n_features": len(feat_cols),
        "train_mean_anomaly_score": float(train_scores.mean()),
        "train_min_anomaly_score": float(train_scores.min())
    }

    if X_test_scaled is not None:
        test_scores = iso.score_samples(X_test_scaled)
        test_preds = iso.predict(X_test_scaled)
        
        # Count anomalies in test
        anomalies_detected = int((test_preds == -1).sum())
        
        # Check distress alignment if `is_distress_status` exists
        if "is_distress_status" in test_df.columns:
            distress_mask = test_df["is_distress_status"] == 1
            if distress_mask.any():
                distress_anomalies = int((test_preds[distress_mask] == -1).sum())
                results["test_distress_rows"] = int(distress_mask.sum())
                results["test_distress_anomalies_flagged"] = distress_anomalies
        
        results["test_mean_anomaly_score"] = float(test_scores.mean())
        results["test_min_anomaly_score"] = float(test_scores.min())
        results["test_total_anomalies"] = anomalies_detected
        results["test_anomaly_rate"] = anomalies_detected / len(test_preds)

    return results

def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline Anomaly Model Training")
    ap.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/models"))
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    train_path = args.features_dir / "joined_train_features.parquet"
    test_path = args.features_dir / "joined_test_features.parquet"

    if not train_path.exists():
        print(f"Error: {train_path} not found. Run multimodal_join.py first.")
        return

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path) if test_path.exists() else pd.DataFrame()

    if "hw_config" not in train_df.columns:
        print("Error: 'hw_config' column missing from joined features.")
        return

    hw_configs = train_df["hw_config"].unique()
    print(f"\nFound {len(hw_configs)} hardware configs: {list(hw_configs)}")

    all_results = {}

    for hw in hw_configs:
        print(f"\n=== Training Model for {hw} ===")
        hw_train = train_df[train_df["hw_config"] == hw].copy()
        hw_test = test_df[test_df["hw_config"] == hw].copy() if not test_df.empty else pd.DataFrame()
        
        res = train_and_evaluate(hw_train, hw_test, hw)
        all_results[hw] = res
        
        if res["status"] == "success":
            print(f"  Train Rows: {res['train_rows']} | Test Rows: {res['test_rows']}")
            if "test_total_anomalies" in res:
                print(f"  Test Anomalies Detected: {res['test_total_anomalies']} ({res['test_anomaly_rate']*100:.2f}%)")
                if "test_distress_rows" in res:
                    print(f"  Captured {res.get('test_distress_anomalies_flagged', 0)} / {res['test_distress_rows']} marked distress states.")

    # Save summary
    summary_path = args.out_dir / "baseline_model_results.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBaseline results saved to {summary_path}")

if __name__ == "__main__":
    main()
