import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def safe_div(n, d):
    return float(n / d) if d > 0 else 0.0

def main():
    print("Loading multimodel_results.json...")
    with open("artifacts/models/multimodel_results.json", "r") as f:
        results = json.load(f)
        
    metrics = {
        "precision_recall_f1": {},
        "inference_latency_ms": {},
        "robustness_test": {}
    }
    
    # 1. Calculate Precision, Recall, F1
    for hw, data in results.items():
        if "models" not in data: continue
        total_distress = data.get("test_distress_rows", 0)
        metrics["precision_recall_f1"][hw] = {}
        
        for model_name, m_data in data["models"].items():
            test_anomalies = m_data.get("test_anomalies", 0)
            caught_distress = m_data.get("caught_distress", 0)
            
            tp = caught_distress
            fp = test_anomalies - caught_distress
            fn = total_distress - caught_distress
            
            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = safe_div(2 * precision * recall, precision + recall)
            
            metrics["precision_recall_f1"][hw][model_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
    print("Precision/Recall/F1 calculated.")
    
    # 2 & 3. Proxy Autoencoder Training for Latency and Robustness
    print("\nLoading feature parquets for Latency and Robustness tests...")
    train_df = pd.read_parquet("artifacts/features/joined_train_features.parquet")
    test_df = pd.read_parquet("artifacts/features/joined_test_features.parquet")
    
    cluster_thresholds = {
        "farm14": 99,
        "farm16": 99,
        "farm18": 99,
        "farm19": 80,
        "farm23": 99
    }
    
    meta_cols = ["node", "timestamp", "hw_config", "is_distress_status", "status", "job"]
    hw_configs = train_df["hw_config"].unique()
    
    for hw in hw_configs:
        print(f"--- Processing {hw} ---")
        hw_train = train_df[train_df["hw_config"] == hw].copy()
        hw_test = test_df[test_df["hw_config"] == hw].copy()
        
        if hw_test.empty:
            print(f"Skipping {hw} (no test data)")
            continue
            
        feat_cols = [c for c in hw_train.columns if c not in meta_cols]
        for c in feat_cols:
            if c not in hw_test.columns:
                hw_test[c] = 0.0
                
        X_train = hw_train[feat_cols].values
        X_test = hw_test[feat_cols].values
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        print("  Training proxy Autoencoder...")
        ae = MLPRegressor(
            hidden_layer_sizes=(64,),
            activation='relu',
            solver='adam',
            batch_size=2048,
            max_iter=20,
            random_state=42,
            early_stopping=True
        )
        ae.fit(X_train_s, X_train_s)
        
        print("  Measuring latency...")
        # Inference Latency
        start_t = time.time()
        _ = ae.predict(X_test_s)
        end_t = time.time()
        
        latency_ms_per_row = ((end_t - start_t) * 1000) / len(X_test_s)
        metrics["inference_latency_ms"][hw] = latency_ms_per_row
        
        print(f"  Latency: {latency_ms_per_row:.4f} ms/row")
        
        # Robustness Test
        print("  Running robustness test...")
        # Add N(0, 0.01) noise
        noise = np.random.normal(0, 0.01, X_test_s.shape)
        X_test_noisy = X_test_s + noise
        
        train_preds = ae.predict(X_train_s)
        train_err = np.mean(np.square(X_train_s - train_preds), axis=1)
        
        test_preds_noisy = ae.predict(X_test_noisy)
        test_err_noisy = np.mean(np.square(X_test_noisy - test_preds_noisy), axis=1)
        
        pct = cluster_thresholds.get(hw, 99)
        thresh = np.percentile(train_err, pct)
        
        if "is_distress_status" in hw_test.columns:
            distress_mask = hw_test["is_distress_status"] == 1
            total_distress = distress_mask.sum()
            
            if total_distress > 0:
                caught = (test_err_noisy[distress_mask] > thresh).sum()
                robust_recall = caught / total_distress
                passed = robust_recall >= 0.90
            else:
                robust_recall = None
                passed = None
                
            metrics["robustness_test"][hw] = {
                "noise_std": 0.01,
                "robust_recall": float(robust_recall) if robust_recall is not None else None,
                "passed_90_pct": bool(passed) if passed is not None else None,
                "distress_rows": int(total_distress)
            }
        else:
            metrics["robustness_test"][hw] = {"status": "No distress column"}

    out_path = Path("artifacts/models/evaluation_report_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nSaved evaluation metrics to {out_path}")

if __name__ == '__main__':
    main()
