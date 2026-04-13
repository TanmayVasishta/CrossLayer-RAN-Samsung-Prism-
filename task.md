# Task: Address Review Feedback (Fix Data Leakage & Add Time-Series EDA)

## Phase 1 — Separate Structural Cleaning ([clean.py](file:///d:/SamsungPrismCrosslayerRAN/eda/clean.py))
- [x] Remove `_iqr_outlier_flag` logic from [clean.py](file:///d:/SamsungPrismCrosslayerRAN/eda/clean.py)
- [x] Restrict `clean_file` to structure only: drop `Unnamed: 0`, parse timestamps, drop NaT/NaN, and deduplicate
- [x] Output `structural_clean.parquet` instead of fully `cleaned_data`
- [x] Update CLI arguments and documentation

## Phase 2 — Explicit Chronological Splitting ([make_splits.py](file:///d:/SamsungPrismCrosslayerRAN/eda/make_splits.py))
- [x] Add explicit print statements confirming split is chronological (time-series safe, no shuffle)
- [x] Make the gap warning threshold (10%) configurable via CLI
- [x] Update JSON output annotations to clarify split type

## Phase 3 — Train-Only Statistics Computation ([compute_train_stats.py](file:///d:/SamsungPrismCrosslayerRAN/eda/compute_train_stats.py))
- [x] Create a new script to read `structural_clean.parquet` files
- [x] Filter rows using the `train` split boundaries from [time_splits.json](file:///d:/SamsungPrismCrosslayerRAN/artifacts/splits/time_splits.json)
- [x] Compute IQR bounds (Q1, Q3) and normalization parameters (mean, std) strictly on `train` data
- [x] Save computed statistics to `train_stats.json`

## Phase 4 — Apply Statistics Globally ([apply_stats.py](file:///d:/SamsungPrismCrosslayerRAN/eda/apply_stats.py))
- [x] Create a script that reads `structural_clean.parquet` and `train_stats.json`
- [x] Add `is_outlier` flag column using the *train-derived* IQR bounds
- [x] Add normalized/standardized value column using *train-derived* mean/std
- [x] Add `dataset_versioning.json` manifest output
- [x] Export final versioned `.parquet` dataset

## Phase 5 — Time-Series EDA ([ts_analysis.py](file:///d:/SamsungPrismCrosslayerRAN/eda/ts_analysis.py))
- [x] Create time-series analysis module
- [x] Load the `train` split data and aggregate (resample) it to a standard frequency (e.g., 5-minute or 15-minute)
- [x] Compute ADF (Augmented Dickey-Fuller) Stationarity test
- [x] Generate ACF (Autocorrelation) and PACF plots for feature lag discovery
- [x] Compute cross-folder correlation metrics (CPU ↔ Memory ↔ Disk)

## Phase 6 — Testing with Real Data
- [x] Update [tests/test_clean_and_lib.py](file:///d:/SamsungPrismCrosslayerRAN/tests/test_clean_and_lib.py)
- [x] Add `test_real_sample_cleaning` that runs the structural clean on a small real `.csv.gz` sample
- [x] Assert `drop_rate < 0.05` and `duplicates == 0` on real data

## Phase 7 — Final Run & Verification
- [x] Run the complete unified pipeline in sequence
- [x] Verify no data leakage in resulting final datasets
