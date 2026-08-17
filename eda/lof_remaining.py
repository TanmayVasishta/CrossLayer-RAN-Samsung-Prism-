"""
eda/lof_remaining.py
====================
Trains ONLY the missing LOF models (cpu_lof_farm19/23 + all disk_lof_*),
builds cpu_lof_scores.parquet + disk_lof_scores.parquet,
evaluates against SLURM ground truth, and writes v7 JSON files.

Existing cpu_lof_farm14/16/18.joblib are loaded, not retrained.
No existing files are overwritten.
"""
from __future__ import annotations
import copy, gc, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS   = Path("artifacts/models")
FEATURES = Path("artifacts/features")
SLURM_TEST = FEATURES / "slurm_data_test_features.parquet"

FARMS = ["farm14", "farm16", "farm18", "farm19", "farm23"]
LOF_NEIGHBORS = 20
LOF_CONTAM    = 0.05
LOF_MAX_TRAIN = 50_000   # subsample cap for LOF.fit()
TOP_K_PCT     = 0.10


# ─── Utilities ─────────────────────────────────────────────────────────────────

def safe_div(a, b, default=0.0):
    return round(a / b, 6) if b > 0 else default

def compute_metrics(y_true, y_pred, scores):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec  = safe_div(tp, tp + fp)
    rec   = safe_div(tp, tp + fn)
    f1    = safe_div(2 * prec * rec, prec + rec)
    fpr   = safe_div(fp, fp + tn)
    k     = max(1, int(len(scores) * TOP_K_PCT))
    top_m = np.zeros(len(scores), dtype=bool)
    top_m[np.argsort(scores)[-k:]] = True
    rr    = safe_div(int((top_m & (y_true == 1)).sum()), tp + fn)
    return {"precision": round(prec,6), "recall": round(rec,6),
            "f1_score": round(f1,6), "false_positive_rate": round(fpr,6),
            "robust_recall": round(rr,6), "latency_ms_per_row": 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "total_flagged": int(y_pred.sum()), "total_distress": int(y_true.sum())}

def fill_and_transform(df, fcols, sc, pca):
    """Scale + PCA-transform a DataFrame slice. Returns PCA array."""
    X = df[fcols].copy()
    for c in fcols:
        med = X[c].median()
        X[c] = X[c].fillna(med if pd.notna(med) else 0.0)
    Xs = sc.transform(X.to_numpy(dtype=float))
    return pca.transform(Xs)

def build_gt(scores_te: pd.DataFrame, slurm: pd.DataFrame) -> pd.DataFrame:
    """Join test scores with SLURM distress ground truth on hw_config+node+minute."""
    s = scores_te.copy()
    s["node"] = s["instance"].str.split(":").str[0]
    s["ts_min"] = pd.to_datetime(s["timestamp"]).dt.floor("1min")
    sl = slurm[["node","hw_config","timestamp","is_distress_status"]].copy()
    sl["timestamp"] = pd.to_datetime(sl["timestamp"])
    sl["ts_min"] = sl["timestamp"].dt.floor("1min")
    agg = sl.groupby(["hw_config","node","ts_min"],sort=False)["is_distress_status"].max().reset_index()
    m = s.merge(agg, on=["hw_config","node","ts_min"], how="left")
    m["is_distress_status"] = m["is_distress_status"].fillna(0).astype(int)
    return m


# ─── Per-farm LOF training ──────────────────────────────────────────────────

def train_one_farm_lof(modality: str, hw: str, df_tr, df_te, fcols) -> dict:
    """
    Load existing scaler+PCA, fit or load LOF, score train+test.
    Returns dict with train_rows, test_rows, lof_path, tr_df, te_df.
    """
    sc  = joblib.load(MODELS / f"{modality}_scaler_{hw}.joblib")
    pca = joblib.load(MODELS / f"{modality}_pca_{hw}.joblib")
    lof_path = MODELS / f"{modality}_lof_{hw}.joblib"

    gtr = df_tr[df_tr["hw_config"] == hw].copy()
    gte = df_te[df_te["hw_config"] == hw].copy()

    print(f"\n  [{hw}] train={len(gtr):,}  test={len(gte):,}")

    Xp_tr = fill_and_transform(gtr, fcols, sc, pca)
    Xp_te = fill_and_transform(gte, fcols, sc, pca) if len(gte) > 0 else np.array([])
    print(f"  [{hw}] PCA dims: {Xp_tr.shape[1]}")

    if lof_path.exists():
        print(f"  [{hw}] Loading existing LOF → {lof_path.name}")
        lof = joblib.load(lof_path)
    else:
        # Subsample for fit
        if len(Xp_tr) > LOF_MAX_TRAIN:
            idx = np.random.default_rng(42).choice(len(Xp_tr), LOF_MAX_TRAIN, replace=False)
            Xp_fit = Xp_tr[idx]
            print(f"  [{hw}] Subsampled {len(Xp_tr):,} → {LOF_MAX_TRAIN:,} for LOF fit")
        else:
            Xp_fit = Xp_tr
            print(f"  [{hw}] Using full {len(Xp_tr):,} rows for LOF fit")

        t0 = time.perf_counter()
        lof = LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, contamination=LOF_CONTAM,
                                 novelty=True, algorithm="ball_tree", leaf_size=30, n_jobs=-1)
        lof.fit(Xp_fit)
        print(f"  [{hw}] LOF fit in {time.perf_counter()-t0:.1f}s")
        joblib.dump(lof, lof_path)
        print(f"  [{hw}] ✓ Saved: {lof_path.name}")

    # Score
    t0 = time.perf_counter()
    tr_lof_score = -lof.score_samples(Xp_tr)
    tr_lof_flag  = lof.predict(Xp_tr) == -1
    if Xp_te.shape[0] > 0:
        te_lof_score = -lof.score_samples(Xp_te)
        te_lof_flag  = lof.predict(Xp_te) == -1
    else:
        te_lof_score = te_lof_flag = np.array([])
    print(f"  [{hw}] Scored in {time.perf_counter()-t0:.1f}s  "
          f"train_flagged={tr_lof_flag.mean()*100:.1f}%  "
          f"test_flagged={te_lof_flag.mean()*100:.1f}%")

    meta = ["instance","timestamp","hw_config"]
    tr_out = gtr[meta].copy().reset_index(drop=True)
    tr_out["lof_score"] = tr_lof_score
    tr_out["lof_flag"]  = tr_lof_flag
    tr_out["split"] = "train"

    te_out = pd.DataFrame()
    if len(gte) > 0:
        te_out = gte[meta].copy().reset_index(drop=True)
        te_out["lof_score"] = te_lof_score
        te_out["lof_flag"]  = te_lof_flag
        te_out["split"] = "test"

    return {"hw": hw, "tr_df": tr_out, "te_df": te_out,
            "pca_components": int(Xp_tr.shape[1])}


def run_modality_lof(modality: str, slurm: pd.DataFrame) -> dict:
    """Full LOF pipeline for one modality. Returns farm_metrics dict."""
    print(f"\n{'='*60}")
    print(f"  LOF [{modality.upper()}]")
    print(f"{'='*60}")

    tr_feat = pd.read_parquet(str(FEATURES / f"{modality}_data_train_features.parquet"))
    te_feat = pd.read_parquet(str(FEATURES / f"{modality}_data_test_features.parquet"))
    print(f"  Train: {tr_feat.shape}  Test: {te_feat.shape}")

    excl  = {"instance","timestamp","hw_config"}
    fcols = [c for c in tr_feat.columns if c not in excl
             and pd.api.types.is_numeric_dtype(tr_feat[c])
             and c in te_feat.columns]
    print(f"  Feature cols: {len(fcols)}")

    all_tr, all_te = [], []
    summary = {}

    for hw in FARMS:
        result = train_one_farm_lof(modality, hw, tr_feat, te_feat, fcols)
        all_tr.append(result["tr_df"])
        if not result["te_df"].empty:
            all_te.append(result["te_df"])
        summary[hw] = {"pca_components": result["pca_components"]}
        gc.collect()

    # Free feature DataFrames now to save RAM before parquet write
    del tr_feat, te_feat
    gc.collect()

    # Combine and save score parquets
    scores_path = MODELS / f"{modality}_lof_scores.parquet"
    if scores_path.exists():
        print(f"\n  ⚠ {scores_path.name} already exists — skipping write")
        all_df = pd.read_parquet(str(scores_path))
    else:
        all_df = pd.concat(all_tr + all_te, ignore_index=True)
        all_df.to_parquet(str(scores_path), index=False, compression="snappy")
        print(f"\n  ✓ Saved: {scores_path}  ({len(all_df):,} rows)")

    # Evaluate on test set
    te_df = all_df[all_df["split"] == "test"].copy()
    te_df["timestamp"] = pd.to_datetime(te_df["timestamp"])
    merged = build_gt(te_df, slurm)
    print(f"\n  Eval joined: {len(merged):,} rows  "
          f"distress={merged['is_distress_status'].sum():,}")

    farm_metrics = {}
    print(f"\n  {'Farm':<8} {'P':>7} {'R':>7} {'F1':>7} {'FPR':>7}")
    for hw in FARMS:
        g = merged[merged["hw_config"] == hw]
        if g.empty:
            continue
        y   = g["is_distress_status"].values.astype(int)
        sc  = g["lof_score"].values
        fl  = g["lof_flag"].values.astype(int)
        m   = compute_metrics(y, fl, sc)
        print(f"  {hw:<8} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['f1_score']:>7.4f} {m['false_positive_rate']:>7.4f}")
        farm_metrics[hw] = {
            "modality": modality, "test_rows": int(len(g)),
            "test_distress_rows": int(y.sum()), "models": {"LOF": m}
        }

    return farm_metrics


# ─── v7 merge ──────────────────────────────────────────────────────────────────

def build_v7(cpu_adp_path, disk_adp_path, cpu_lof_fm, disk_lof_fm):
    print(f"\n{'='*60}")
    print("  BUILDING v7 JSONs")
    print(f"{'='*60}")

    with open(MODELS / "evaluation_report_metrics_v6.json") as f:
        v6_eval = json.load(f)
    with open(MODELS / "multimodel_results_v6.json") as f:
        v6_multi = json.load(f)
    with open(cpu_adp_path) as f:
        cpu_adp = json.load(f)
    with open(disk_adp_path) as f:
        disk_adp = json.load(f)

    v7_eval  = copy.deepcopy(v6_eval)
    v7_multi = copy.deepcopy(v6_multi)
    v7_eval["generated_at"]  = datetime.now().isoformat(timespec="seconds")
    v7_multi["generated_at"] = datetime.now().isoformat(timespec="seconds")
    v7_eval["note"]  = "v7: AE_Adaptive + Ens_Adaptive + LOF for CPU and Disk"
    v7_multi["note"] = v7_eval["note"]

    def inject_eval(mod, farm_metrics, tag=None):
        dest = v7_eval.setdefault("by_modality", {}).setdefault(mod, {})
        for hw, fd in farm_metrics.items():
            dest.setdefault(hw, {})
            for mname, m in fd["models"].items():
                key = tag or mname
                dest[hw][key] = {k: m[k] for k in
                    ["precision","recall","f1_score","false_positive_rate",
                     "robust_recall","latency_ms_per_row"]}

    def inject_multi(mod, farm_metrics, tag=None):
        dest = v7_multi.setdefault("by_modality", {}).setdefault(mod, {})
        for hw, fd in farm_metrics.items():
            if hw not in dest:
                dest[hw] = {"status":"success","modality":mod,
                            "test_rows": fd.get("test_rows",0),
                            "test_distress_rows": fd.get("test_distress_rows",0)}
            for mname, m in fd["models"].items():
                key = tag or mname
                dest[hw][key] = {"test_anomalies": m["total_flagged"],
                                 "caught_distress": m["tp"]}

    # Inject adaptive metrics (from saved JSON)
    for mod, adp in [("cpu", cpu_adp), ("disk", disk_adp)]:
        fm = adp.get("farm_metrics", {})
        inject_eval(mod, fm)
        inject_multi(mod, fm)

    # Inject LOF metrics
    inject_eval("cpu",  cpu_lof_fm,  "LOF")
    inject_multi("cpu", cpu_lof_fm,  "LOF")
    inject_eval("disk", disk_lof_fm, "LOF")
    inject_multi("disk",disk_lof_fm, "LOF")

    # Write v7 — never overwrite
    for fname, data in [("evaluation_report_metrics_v7.json", v7_eval),
                        ("multimodel_results_v7.json", v7_multi)]:
        out = MODELS / fname
        if out.exists():
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = MODELS / fname.replace(".json", f"_{ts}.json")
            print(f"  ⚠ Already exists — saving as {out.name}")
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✓ Saved: {out.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LOF REMAINING FARMS + v7 MERGE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    slurm = pd.read_parquet(str(SLURM_TEST))
    print(f"  SLURM test rows: {len(slurm):,}")

    cpu_lof_fm  = run_modality_lof("cpu",  slurm)
    disk_lof_fm = run_modality_lof("disk", slurm)

    build_v7(MODELS / "cpu_adaptive_thr.json",
             MODELS / "disk_adaptive_thr.json",
             cpu_lof_fm, disk_lof_fm)

    print("\n" + "=" * 60)
    print("  DONE — LOF Results Summary")
    print("=" * 60)
    for mod, fm in [("CPU", cpu_lof_fm), ("Disk", disk_lof_fm)]:
        print(f"\n  {mod}:")
        for hw, fd in fm.items():
            m = fd["models"]["LOF"]
            print(f"    {hw}: R={m['recall']:.4f}  P={m['precision']:.4f}  "
                  f"F1={m['f1_score']:.4f}  FPR={m['false_positive_rate']:.4f}")


if __name__ == "__main__":
    main()
