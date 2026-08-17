"""
eda/multimodal_join.py
======================
Joins memory_data_features and slurm_data_features on (node, timestamp).
Provides a unified feature set per node, per minute for the anomaly model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eda.lib import ensure_dir

def join_modalities(mem_df: pd.DataFrame, slurm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge memory and slurm DataFrames.
    - Memory uses 'instance', Slurm uses 'node'
    - Round timestamps to nearest minute
    - Outer join
    - Forward fill missing features (up to 3 min)
    """
    # 1. Align column names and clean node strings
    if "instance" in mem_df.columns:
        mem_df = mem_df.rename(columns={"instance": "node"})
        
    # Strip port from memory node names (e.g. 'farm140151:9100' -> 'farm140151')
    mem_df["node"] = mem_df["node"].astype(str).str.split(":").str[0]
        
    # 2. Floor timestamps to 1-minute
    mem_df["timestamp"] = pd.to_datetime(mem_df["timestamp"]).dt.floor("1min")
    slurm_df["timestamp"] = pd.to_datetime(slurm_df["timestamp"]).dt.floor("1min")

    # 3. Handle duplicates from flooring (keep last)
    mem_df = mem_df.sort_values("timestamp").drop_duplicates(subset=["node", "timestamp"], keep="last")
    slurm_df = slurm_df.sort_values("timestamp").drop_duplicates(subset=["node", "timestamp"], keep="last")

    # Drop hw_config from one of them to prevent conflicts if present in both
    if "hw_config" in mem_df.columns and "hw_config" in slurm_df.columns:
        slurm_df = slurm_df.drop(columns=["hw_config"])

    # 4. Outer join
    joined = pd.merge(
        mem_df,
        slurm_df,
        on=["node", "timestamp"],
        how="outer"
    )

    # 5. Forward fill (limit=3) per node to handle minor scrape misalignments
    joined = joined.sort_values(["node", "timestamp"])
    
    # We must group by node to ffill properly
    cols_to_fill = [c for c in joined.columns if c not in ["node", "timestamp", "hw_config"]]
    joined[cols_to_fill] = joined.groupby("node")[cols_to_fill].ffill(limit=3)

    return joined

def main() -> None:
    ap = argparse.ArgumentParser(description="Multimodal join for memory and slurm features.")
    ap.add_argument("--features-dir", type=Path, default=Path("artifacts/features"),
                    help="Directory containing individual feature parquets")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/features"),
                    help="Output directory for joined parquets")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    for split in ["train", "test"]:
        mem_path = args.features_dir / f"memory_data_{split}_features.parquet"
        slurm_path = args.features_dir / f"slurm_data_{split}_features.parquet"

        if not mem_path.exists() or not slurm_path.exists():
            print(f"  [SKIP] Missing input files for {split} split. Skipping multimodal join.")
            continue

        print(f"\nProcessing '{split}' split...")
        mem_df = pd.read_parquet(mem_path)
        slurm_df = pd.read_parquet(slurm_path)

        print(f"  Memory {split} shape: {mem_df.shape}")
        print(f"  Slurm {split} shape:  {slurm_df.shape}")

        joined_df = join_modalities(mem_df, slurm_df)

        # Drop rows that are predominantly NaN even after ffill
        feat_cols = [c for c in joined_df.columns if c not in ["node", "timestamp", "hw_config"]]
        nan_frac = joined_df[feat_cols].isna().mean(axis=1)
        valid_mask = nan_frac <= 0.50
        
        dropped = (~valid_mask).sum()
        joined_df = joined_df[valid_mask].reset_index(drop=True)
        
        # Reconstruct hw_config safely from node name to guarantee it's populated and a string
        joined_df["hw_config"] = joined_df["node"].astype(str).str[:6]

        # Fill missing string/object columns with 'unknown' to prevent PyArrow type mixing
        obj_cols = joined_df.select_dtypes(include=['object', 'string', 'category']).columns
        for c in obj_cols:
            joined_df[c] = joined_df[c].fillna("unknown").astype(str)

        # Fill remaining NaNs (which are purely numeric now) with 0
        joined_df = joined_df.fillna(0)

        out_path = args.out_dir / f"joined_{split}_features.parquet"
        joined_df.to_parquet(out_path, index=False, compression="snappy")

        print(f"  Joined {split} shape: {joined_df.shape} (dropped {dropped} sparse rows)")
        print(f"  Saved to: {out_path}")

if __name__ == "__main__":
    main()
