"""
eda/clean.py
============
Chunk-safe structural data cleaning pipeline for raw .csv.gz metric files.
(Phase 1 of cleaning)

Steps per chunk:
  1. Drop "Unnamed: 0" (CSV export artifact)
  2. Parse timestamp → pd.Timestamp UTC-safe, drop NaT rows
  3. Cast value → float64, drop NaN value rows
  4. Deduplicate on key columns (timestamp + label cols), keep last
  5. Append structurally cleaned chunk to output Parquet file

NOTE: Statistical flagging (IQR outliers, normalization) is intentionally NOT done here.
It must be done AFTER splitting, using statistics from the TRAIN split only to prevent data leakage.

Usage:
  python -m eda.clean --input-file dataset/cpu_data/some_file.csv.gz --out-dir structural_clean/cpu_data/
  python -m eda.clean --all-folders --dataset-dir dataset --out-dir structural_clean/
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from eda.lib import (
    DEFAULT_TIME_COL_CANDIDATES,
    DEFAULT_VALUE_COL_CANDIDATES,
    _pick_first_existing,
    _to_datetime_safe,
    ensure_dir,
    list_csv_gz_files,
    write_json,
)

_JUNK_COLS = {"Unnamed: 0", "index", "level_0"}
CHUNKSIZE = 250_000


@dataclass
class CleanStats:
    """Accumulates row-level structural cleaning stats for reporting."""
    file: str = ""
    rows_read: int = 0
    rows_dropped_dup: int = 0
    rows_dropped_nat_ts: int = 0
    rows_dropped_nan_value: int = 0
    rows_written: int = 0
    columns_dropped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "rows_read": self.rows_read,
            "rows_dropped_dup": self.rows_dropped_dup,
            "rows_dropped_nat_ts": self.rows_dropped_nat_ts,
            "rows_dropped_nan_value": self.rows_dropped_nan_value,
            "rows_written": self.rows_written,
            "columns_dropped": self.columns_dropped,
            "retention_rate": round(self.rows_written / self.rows_read, 4) if self.rows_read > 0 else None,
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Structural clean summary: {Path(self.file).name}")
        print(f"{'='*60}")
        print(f"  Rows read           : {self.rows_read:>12,}")
        print(f"  Dropped (NaT ts)    : {self.rows_dropped_nat_ts:>12,}")
        print(f"  Dropped (NaN value) : {self.rows_dropped_nan_value:>12,}")
        print(f"  Dropped (duplicates): {self.rows_dropped_dup:>12,}")
        print(f"  Rows written        : {self.rows_written:>12,}")
        if self.columns_dropped:
            print(f"  Columns dropped     : {self.columns_dropped}")
        if self.rows_read > 0:
            ret = round(100 * self.rows_written / self.rows_read, 2)
            print(f"  Retention rate      : {ret}%")
        print(f"{'='*60}")


def structural_clean(
    input_path: Path,
    out_dir: Path,
    chunksize: int = CHUNKSIZE,
    overwrite: bool = False,
) -> CleanStats:
    """
    Structurally clean a single .csv.gz file and write the result as a .parquet file.
    Does NOT compute or apply any statistics to prevent data leakage.
    """
    input_path = Path(input_path)
    ensure_dir(out_dir)
    # Target suffix indicates it's purely structural
    out_path = out_dir / (input_path.stem.replace(".csv", "") + ".parquet")

    stats = CleanStats(file=str(input_path))

    if out_path.exists() and not overwrite:
        print(f"  [skip] {out_path.name} already exists (use --overwrite to redo).")
        return stats

    writer: Optional[pq.ParquetWriter] = None
    schema: Optional[pa.Schema] = None
    time_col: Optional[str] = None
    value_col: Optional[str] = None
    key_cols: Optional[List[str]] = None

    try:
        for chunk in pd.read_csv(input_path, compression="gzip", chunksize=chunksize, low_memory=False):
            stats.rows_read += len(chunk)

            # 1. Drop junk index columns
            junk = [c for c in chunk.columns if c in _JUNK_COLS]
            if junk:
                chunk = chunk.drop(columns=junk)
                if junk not in stats.columns_dropped:
                    stats.columns_dropped.extend(j for j in junk if j not in stats.columns_dropped)

            if time_col is None:
                time_col = _pick_first_existing(chunk.columns, DEFAULT_TIME_COL_CANDIDATES)
                value_col = _pick_first_existing(chunk.columns, DEFAULT_VALUE_COL_CANDIDATES)
                non_metric_cols = [
                    c for c in chunk.columns
                    if c not in {time_col or "", value_col or ""}
                ]
                key_cols = ([time_col] if time_col else []) + non_metric_cols[:5]

            # 2. Parse timestamp → datetime, drop NaT
            if time_col and time_col in chunk.columns:
                chunk[time_col] = _to_datetime_safe(chunk[time_col])
                nat_mask = chunk[time_col].isna()
                n_nat = int(nat_mask.sum())
                if n_nat:
                    stats.rows_dropped_nat_ts += n_nat
                    chunk = chunk[~nat_mask]
            if chunk.empty:
                continue

            # 3. Cast value → float64, drop NaN
            if value_col and value_col in chunk.columns:
                chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
                nan_mask = chunk[value_col].isna()
                n_nan = int(nan_mask.sum())
                if n_nan:
                    stats.rows_dropped_nan_value += n_nan
                    chunk = chunk[~nan_mask]
            if chunk.empty:
                continue

            # 4. Deduplicate
            if key_cols:
                valid_keys = [c for c in key_cols if c in chunk.columns]
                if valid_keys:
                    before = len(chunk)
                    chunk = chunk.drop_duplicates(subset=valid_keys, keep="last")
                    stats.rows_dropped_dup += before - len(chunk)
            if chunk.empty:
                continue

            # Note: We deliberately DO NOT apply IQR outlier limits here to avoid train/test data leakage.
            
            stats.rows_written += len(chunk)

            # 5. Write Parquet chunk
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(str(out_path), schema, compression="snappy")
            else:
                table = table.cast(schema)
            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    return stats


def clean_folder(
    folder_path: Path,
    out_dir: Path,
    chunksize: int = CHUNKSIZE,
    overwrite: bool = False,
    max_files: Optional[int] = None,
) -> List[CleanStats]:
    files = sorted(Path(folder_path).glob("*.csv.gz"))
    if not files:
        print(f"  No .csv.gz files found in {folder_path}")
        return []
        
    if max_files and max_files > 0:
        files = files[:max_files]
        
    all_stats = []
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] Structural clean: {f.name} ...")
        t0 = time.perf_counter()
        st = structural_clean(f, out_dir, chunksize=chunksize, overwrite=overwrite)
        elapsed = time.perf_counter() - t0
        st.print_summary()
        print(f"  Done in {elapsed:.1f}s")
        all_stats.append(st)
    return all_stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 1 Cleaning: structural fixes (deduplication, dtypes) for .csv.gz -> .parquet. No stats applied."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=Path, help="Single .csv.gz file to structurally clean")
    group.add_argument("--folder", type=Path, help="Folder of .csv.gz files to structurally clean")
    group.add_argument("--all-folders", action="store_true", help="Process all nested metric folders in dataset/")
    
    ap.add_argument("--out-dir", type=Path, default=Path("structural_clean"), help="Output directory for .parquet files")
    ap.add_argument("--dataset-dir", type=Path, default=Path("dataset"), help="Dataset root (used with --all-folders)")
    ap.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    ap.add_argument("--max-files", type=int, default=0, help="Max files per folder (0 for unlimited)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    ap.add_argument("--stats-out", type=Path, default=None, help="Optional JSON path to write structural cleaning stats")
    args = ap.parse_args()

    all_stats: List[CleanStats] = []

    if args.input_file:
        st = structural_clean(args.input_file, args.out_dir, chunksize=args.chunksize, overwrite=args.overwrite)
        st.print_summary()
        all_stats.append(st)

    elif args.folder:
        all_stats = clean_folder(args.folder, args.out_dir, chunksize=args.chunksize, overwrite=args.overwrite, max_files=args.max_files)

    elif args.all_folders:
        file_map = list_csv_gz_files(args.dataset_dir)
        for folder_name, files in file_map.items():
            print(f"\n{'#'*60}")
            if args.max_files and args.max_files > 0:
                files = files[:args.max_files]
            print(f"  Folder: {folder_name}  ({len(files)} files)")
            print(f"{'#'*60}")
            folder_out = args.out_dir / folder_name
            for i, f in enumerate(files, 1):
                print(f"  [{i}/{len(files)}] {f.name} ...")
                t0 = time.perf_counter()
                st = structural_clean(f, folder_out, chunksize=args.chunksize, overwrite=args.overwrite)
                st.print_summary()
                print(f"  Done in {time.perf_counter() - t0:.1f}s")
                all_stats.append(st)

    stats_target = args.stats_out or (args.out_dir / "structural_cleaning_stats.json")
    write_json(stats_target, [s.to_dict() for s in all_stats])
    print(f"\nStructural cleaning stats written to: {stats_target}")


if __name__ == "__main__":
    main()
