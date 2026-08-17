"""
eda/ts_analysis.py
==================
Time-Series Exploratory Data Analysis on the TRAIN SPLIT (normal data only).

Updated for binary anomaly-detection split:
  - Train boundary: May 19 00:00 → May 22 23:59 (hardcoded, aligned with make_splits.py)
  - Uses per-modality resample frequencies matching the split pipeline
  - cpu_data    : 5T  → ADF + ACF/PACF
  - disk_data   : 5T  → ADF + ACF/PACF
  - memory_data : 1T  → ADF + ACF/PACF
  - slurm_data  : skipped (event-driven, not resampled)

Outputs:
  reports/ts/ts_analysis_summary.json
  reports/ts/{modality}_sample_acf.png
  reports/ts/{modality}_sample_pacf.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Headless backend

from eda.lib import ensure_dir, write_json

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AV = True
except ImportError:
    STATSMODELS_AV = False

# Aligned with make_splits.py
TRAIN_START = pd.Timestamp("2023-05-19 00:00:00")
TRAIN_END   = pd.Timestamp("2023-05-22 23:59:59.999999999")

# Per-modality resample — must match make_splits.py RESAMPLE_RULES
MODALITY_RESAMPLE: Dict[str, Optional[str]] = {
    "cpu_data":    "5T",
    "disk_data":   "5T",
    "memory_data": "1T",
    "slurm_data":  None,   # event-driven — skip TS analysis
}


def _safe_adf(series: pd.Series) -> Dict[str, Any]:
    if not STATSMODELS_AV:
        return {"error": "statsmodels not installed"}
    if len(series) < 30:
        return {"error": f"Too few samples after resampling ({len(series)} points)"}
    if series.nunique() <= 1:
        return {"error": "Constant series — cannot test stationarity"}

    try:
        res = adfuller(series.dropna(), maxlag=30, autolag="AIC")
        adf_stat  = float(res[0])
        p_value   = float(res[1])
        crit_vals = {k: round(float(v), 4) for k, v in dict(res[4]).items()}
        return {
            "adf_statistic":    round(adf_stat, 4),
            "p_value":          round(p_value, 6),
            "critical_values":  crit_vals,
            "is_stationary_95": bool(p_value < 0.05),
            "interpretation": (
                "Stationary at 95% — suitable for ARIMA / LSTM without differencing."
                if p_value < 0.05
                else "Non-stationary at 95% — consider differencing or detrending before modelling."
            ),
        }
    except Exception as e:
        return {"error": str(e)}


def generate_acf_pacf_plots(
    series: pd.Series,
    out_prefix: Path,
    title: str,
    n_lags: int = 40,
) -> None:
    if not STATSMODELS_AV:
        return
    series = series.dropna()
    if len(series) < 50:
        print(f"  [SKIP] Too few points ({len(series)}) for ACF/PACF plots")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(series, ax=ax, lags=min(n_lags, len(series) // 2 - 1), title=f"ACF: {title}")
    fig.tight_layout()
    acf_path = str(out_prefix) + "_acf.png"
    fig.savefig(acf_path, dpi=120)
    plt.close(fig)
    print(f"  Saved ACF plot: {acf_path}")

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_pacf(series, ax=ax, lags=min(n_lags, len(series) // 2 - 1),
              title=f"PACF: {title}", method="ywm")
    fig.tight_layout()
    pacf_path = str(out_prefix) + "_pacf.png"
    fig.savefig(pacf_path, dpi=120)
    plt.close(fig)
    print(f"  Saved PACF plot: {pacf_path}")


def analyse_modality(
    folder_dir: Path,
    modality: str,
    out_dir: Path,
    time_col: str,
    value_col: str,
) -> Dict[str, Any]:
    resample_rule = MODALITY_RESAMPLE.get(modality)
    if resample_rule is None:
        print(f"  [SKIP] {modality} — event-driven, no resampling, TS analysis not applicable.")
        return {"skipped": True, "reason": "event-driven data, no resampling applied"}

    files = sorted(folder_dir.glob("*.parquet"))
    if not files:
        return {"error": f"No parquet files found in {folder_dir}"}

    sample_file = files[0]
    print(f"\n  Analysing {modality} via representative file: {sample_file.name}")
    print(f"  Resample rule: {resample_rule}")

    try:
        df = pd.read_parquet(sample_file, columns=[time_col, value_col])
    except Exception as e:
        return {"error": str(e)}

    # Filter to TRAIN split only
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[(df[time_col] >= TRAIN_START) & (df[time_col] <= TRAIN_END)].dropna(subset=[time_col])

    if df.empty:
        return {"error": f"No rows in train window for {sample_file.name}"}

    # Resample with mean aggregation + forward-fill (max 3)
    df = df.set_index(time_col).sort_index()
    series = df[value_col].resample(resample_rule).mean().ffill(limit=3).dropna()

    print(f"  Train points after resample: {len(series):,}")

    adf_res = _safe_adf(series)
    print(f"  ADF p-value: {adf_res.get('p_value', 'N/A')}  "
          f"Stationary@95%: {adf_res.get('is_stationary_95', 'N/A')}")

    # ACF / PACF plots
    generate_acf_pacf_plots(
        series,
        out_dir / f"{modality}_sample",
        title=f"{modality} — Train Span ({TRAIN_START.date()} → {TRAIN_END.date()})",
    )

    return {
        "representative_file": sample_file.name,
        "resample_freq":       resample_rule,
        "train_start":         str(TRAIN_START),
        "train_end":           str(TRAIN_END),
        "train_points":        int(len(series)),
        "value_range":         [round(float(series.min()), 4), round(float(series.max()), 4)],
        "mean":                round(float(series.mean()), 4),
        "std":                 round(float(series.std()), 4),
        "stationarity_adf":    adf_res,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Time-series EDA on TRAIN split (normal data only — May 19-22)."
    )
    ap.add_argument("--clean-dir", type=Path, default=Path("structural_clean"))
    ap.add_argument("--out-dir",   type=Path, default=Path("reports/ts"))
    ap.add_argument("--time-col",  type=str, default="timestamp")
    ap.add_argument("--value-col", type=str, default="value")
    args = ap.parse_args()

    if not STATSMODELS_AV:
        print("[ERROR] statsmodels is not installed. Run: pip install statsmodels matplotlib")
        sys.exit(1)

    ensure_dir(args.out_dir)
    print(f"TS Analysis — Train window: {TRAIN_START} → {TRAIN_END}")

    results: Dict[str, Any] = {}
    for modality, rule in MODALITY_RESAMPLE.items():
        folder_dir = args.clean_dir / modality
        if not folder_dir.exists():
            print(f"\n[SKIP] {folder_dir} not found")
            continue
        print(f"\n{'='*60}")
        print(f"  Modality: {modality}")
        print(f"{'='*60}")
        results[modality] = analyse_modality(
            folder_dir=folder_dir,
            modality=modality,
            out_dir=args.out_dir,
            time_col=args.time_col,
            value_col=args.value_col,
        )

    out_json = args.out_dir / "ts_analysis_summary.json"
    write_json(out_json, results)
    print(f"\nTS summary → {out_json}")


if __name__ == "__main__":
    main()
