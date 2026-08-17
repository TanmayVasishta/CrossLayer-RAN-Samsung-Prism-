"""
Utility: re-save CPU autoencoder joblib files so they reference
models.autoencoder.ReconAE instead of eda.cpu_pipeline_v3.ReconAE.

Run once from project root:
    python fix_ae_pickle.py
"""
import sys
sys.path.insert(0, '.')

import joblib

# Step 1: make eda.cpu_pipeline_v3.ReconAE importable so old files can load
from eda.cpu_pipeline_v3 import ReconAE  # noqa: F401 — needed for unpickling

# Step 2: import the new canonical location
from models.autoencoder import ReconAE as CanonicalReconAE

farms = ['farm14', 'farm16', 'farm18', 'farm19', 'farm23']

for farm in farms:
    path = f'artifacts/models/cpu_autoencoder_{farm}.joblib'

    # Load using old class (eda.cpu_pipeline_v3.ReconAE in scope)
    old_ae = joblib.load(path)

    # Create fresh instance of canonical class, copy all state
    new_ae = CanonicalReconAE.__new__(CanonicalReconAE)
    new_ae.__dict__.update(old_ae.__dict__)

    # Verify it scores correctly
    import numpy as np
    dummy = np.zeros((10, 182))
    scores, flags = new_ae.score(dummy)
    assert len(scores) == 10, "score() returned wrong length"

    # Re-save with new class path
    joblib.dump(new_ae, path, compress=3)
    print(f'  OK  {path}')

print('\nAll 5 CPU autoencoders re-serialized as models.autoencoder.ReconAE')
