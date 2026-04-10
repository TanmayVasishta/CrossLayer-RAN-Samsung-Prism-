from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from eda.lib import FileScanConfig, scan_csv_gz, list_csv_gz_files
from eda.clean import structural_clean


def test_scan_csv_gz_synthetic(tmp_path):
    # Synthetic basic test for liberal structural columns
    csv_file = tmp_path / "test_data.csv.gz"
    
    data = {
        "Unnamed: 0": [0, 1, 2, 3, 4, 5, 6],
        "timestamp": [
            "2023-01-01 00:00:00",
            "2023-01-01 00:00:10",
            "2023-01-01 00:00:20",
            None,  # NaT
            "2023-01-01 00:00:40",
            "2023-01-01 00:00:50",
            "2023-01-01 00:00:10",  # Duplicate row time
        ],
        "node": ["node1", "node1", "node1", "node1", "node1", "node1", "node1"],
        "value": [
            10.0, 
            12.0, 
            11.0, 
            15.0, 
            None, # NaN
            9999.0, # Outlier
            12.0 # Duplicate row value
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False, compression="gzip")
    
    cfg = FileScanConfig()
    res = scan_csv_gz(csv_file, cfg)
    
    assert "Unnamed: 0" not in res["columns"], "Junk col should be dropped"
    assert res["duplicate_row_count"] == 1, "Should find 1 exact duplicate"


def test_real_sample_structural_cleaning(tmp_path):
    """
    Sanity check on TRUE DATA from the repository.
    Verifies that the new separated structural cleaner:
    1) Works end-to-end on a real chunk
    2) Permits exactly 0 duplicates in output Parquet
    3) Does not obliterate >5% of the data in structural passes
    """
    ds_dir = Path("dataset")
    if not ds_dir.exists():
        pytest.skip("No dataset directory found. Skipping real-data integration test.")
        
    ds_files = list_csv_gz_files(ds_dir)
    real_target = None
    for folder, files in ds_files.items():
        if files:
            real_target = files[0] # Grab the first real file we can find
            break
            
    if not real_target:
        pytest.skip("No actual .csv.gz files found in dataset to test against.")
        
    out_dir = tmp_path / "real_clean_out"
    
    # Run the structural clean
    # Note: Using small chunksize to simulate chunk logic on a smaller scale
    stats = structural_clean(input_path=real_target, out_dir=out_dir, chunksize=5000)
    
    pq_path = out_dir / (real_target.stem.replace(".csv", "") + ".parquet")
    assert pq_path.exists(), "Structural cleaner failed to produce .parquet output"
    
    # 1) Verify Row Drop Tolerance
    raw_count = stats.rows_read
    valid_count = stats.rows_written
    
    assert raw_count > 0, "Test file was empty"
    drop_rate = 1.0 - (valid_count / raw_count)
    assert drop_rate < 0.05, f"Lost {drop_rate*100:.2f}% of rows during struct clean, exceeding 5% tolerance max!"
    
    # 2) Verify deduplication functionally worked across the Parquet file
    df_clean = pd.read_parquet(pq_path)
    
    # Identify key cols (assuming 'timestamp' and labels that aren't 'value' or 'Unnamed: 0')
    time_col = "timestamp" if "timestamp" in df_clean.columns else None
    val_col = "value" if "value" in df_clean.columns else None
    
    if time_col and val_col:
        key_cols = [c for c in df_clean.columns if c != val_col]
        n_dups = df_clean.duplicated(subset=key_cols).sum()
        assert n_dups == 0, f"Found {n_dups} surviving duplicates in {real_target.name} after cleaning!"

