from __future__ import annotations

import dataclasses
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_TIME_COL_CANDIDATES = ("timestamp", "time", "ts")
DEFAULT_VALUE_COL_CANDIDATES = ("value", "val")


@dataclass
class FileScanConfig:
    chunksize: int = 250_000
    sample_rows: int = 200_000
    categorical_topk: int = 20
    parse_timestamps: bool = True


def list_csv_gz_files(dataset_dir: Path) -> Dict[str, List[Path]]:
    dataset_dir = Path(dataset_dir)
    folders = ["cpu_data", "disk_data", "memory_data", "slurm_data"]
    out: Dict[str, List[Path]] = {}
    for folder in folders:
        p = dataset_dir / folder
        if not p.exists():
            continue
        out[folder] = sorted(p.glob("*.csv.gz"))
    return out


def _pick_first_existing(columns: Iterable[str], candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _to_datetime_safe(s: pd.Series) -> pd.Series:
    # Handles nanosecond-ish strings like "2023-05-19 04:00:52.737999872"
    # format='ISO8601' is 100x faster than dateutil fallback
    return pd.to_datetime(s, errors="coerce", utc=False, format="ISO8601")


def _reservoir_add(existing: np.ndarray, new_values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Keep up to k values, mixing in new_values uniformly.
    Simple approach: if underfull, append; else random replace.
    """
    if k <= 0:
        return existing
    if existing.size < k:
        take = min(k - existing.size, new_values.size)
        if take > 0:
            existing = np.concatenate([existing, new_values[:take]])
        new_values = new_values[take:]
    if new_values.size == 0 or existing.size == 0:
        return existing
    # Randomly replace elements in existing with some from new_values
    # Probability proportional; approximate with random indices.
    n_replace = min(new_values.size, max(1, k // 20))
    idx = rng.integers(0, existing.size, size=n_replace, endpoint=False)
    pick = rng.choice(new_values, size=n_replace, replace=False) if new_values.size >= n_replace else new_values
    existing[idx[: pick.size]] = pick
    return existing


def scan_csv_gz(path: Path, cfg: FileScanConfig) -> Dict[str, Any]:
    path = Path(path)
    rng = np.random.default_rng(12345)

    row_count = 0
    missing_by_col: Counter[str] = Counter()
    columns: List[str] = []
    categorical_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    ts_min: Optional[pd.Timestamp] = None
    ts_max: Optional[pd.Timestamp] = None
    ts_deltas_sample: List[float] = []

    value_samples = np.array([], dtype=np.float64)

    time_col = None
    value_col = None

    for chunk in pd.read_csv(path, compression="gzip", chunksize=cfg.chunksize):
        if not columns:
            columns = list(chunk.columns)
            time_col = _pick_first_existing(columns, DEFAULT_TIME_COL_CANDIDATES)
            value_col = _pick_first_existing(columns, DEFAULT_VALUE_COL_CANDIDATES)

        row_count += len(chunk)
        missing_by_col.update({c: int(chunk[c].isna().sum()) for c in chunk.columns})

        # Drop common "Unnamed: 0" index artifacts for categorical counting
        cat_cols = [
            c
            for c in chunk.columns
            if c not in {"Unnamed: 0"} and chunk[c].dtype == "object" and c != (time_col or "")
        ]
        for c in cat_cols:
            vc = chunk[c].astype("string").fillna("<NA>")
            # cap work per chunk
            for k, v in vc.value_counts(dropna=False).head(cfg.categorical_topk).items():
                categorical_counts[c][str(k)] += int(v)

        if cfg.parse_timestamps and time_col and time_col in chunk.columns:
            ts = _to_datetime_safe(chunk[time_col])
            tmin = ts.min()
            tmax = ts.max()
            if pd.notna(tmin):
                ts_min = tmin if ts_min is None else min(ts_min, tmin)
            if pd.notna(tmax):
                ts_max = tmax if ts_max is None else max(ts_max, tmax)

            # sample deltas for approximate cadence
            ts2 = ts.dropna()
            if len(ts2) >= 3:
                s = ts2.sample(n=min(5000, len(ts2)), random_state=1).sort_values()
                deltas = (s.diff().dropna().dt.total_seconds()).values
                if deltas.size:
                    ts_deltas_sample.extend(deltas[:2000].tolist())

        if value_col and value_col in chunk.columns:
            v = pd.to_numeric(chunk[value_col], errors="coerce").dropna()
            if len(v):
                # light sampling from this chunk
                take_n = min(len(v), max(5000, cfg.sample_rows // 40))
                sampled = v.sample(n=take_n, random_state=2).to_numpy(dtype=np.float64, copy=False)
                value_samples = _reservoir_add(value_samples, sampled, cfg.sample_rows, rng)

    value_stats = {}
    if value_samples.size:
        value_stats = {
            "count_sampled": int(value_samples.size),
            "min": float(np.nanmin(value_samples)),
            "max": float(np.nanmax(value_samples)),
            "mean": float(np.nanmean(value_samples)),
            "std": float(np.nanstd(value_samples)),
            "p01": float(np.nanquantile(value_samples, 0.01)),
            "p05": float(np.nanquantile(value_samples, 0.05)),
            "p50": float(np.nanquantile(value_samples, 0.50)),
            "p95": float(np.nanquantile(value_samples, 0.95)),
            "p99": float(np.nanquantile(value_samples, 0.99)),
        }

    cadence = None
    if ts_deltas_sample:
        arr = np.asarray(ts_deltas_sample, dtype=np.float64)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            cadence = {
                "median_seconds": float(np.median(arr)),
                "p05_seconds": float(np.quantile(arr, 0.05)),
                "p95_seconds": float(np.quantile(arr, 0.95)),
            }

    return {
        "path": str(path.as_posix()),
        "file_name": path.name,
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "rows": int(row_count),
        "columns": columns,
        "time_col": time_col,
        "value_col": value_col,
        "missing_by_col": dict(missing_by_col),
        "timestamp_min": None if ts_min is None else str(ts_min),
        "timestamp_max": None if ts_max is None else str(ts_max),
        "cadence": cadence,
        "value_stats": value_stats,
        "categorical_top": {
            c: [{"value": k, "count": int(v)} for k, v in categorical_counts[c].most_common(cfg.categorical_topk)]
            for c in sorted(categorical_counts.keys())
        },
    }


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

