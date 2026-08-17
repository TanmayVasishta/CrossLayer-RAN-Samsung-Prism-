"""
models/autoencoder.py
=====================
Shared ReconAE class (MLPRegressor-based autoencoder for anomaly detection).

Saved here so cpu_autoencoder_*.joblib can be deserialized from ANY script
without needing to import eda.cpu_pipeline_v3.

Usage:
    from models.autoencoder import ReconAE
    ae = joblib.load("artifacts/models/cpu_autoencoder_farm14.joblib")
    scores, flags = ae.score(X)
"""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

AE_SIGMA = 3.0


class ReconAE:
    """MLPRegressor-based reconstruction autoencoder for anomaly scoring."""

    def __init__(self, n: int):
        h1 = min(64, max(n, 4))
        h2 = min(16, max(n // 4, 2))
        self.sc = StandardScaler()
        self.m  = MLPRegressor(
            hidden_layer_sizes=(h1, h2, h1),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=30,
            batch_size=256,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            verbose=False,
        )
        self.thr: float = 0.0
        self.mu:  float = 0.0
        self.sd:  float = 0.0

    def fit(self, X: np.ndarray) -> "ReconAE":
        Xs = self.sc.fit_transform(X)
        self.m.fit(Xs, Xs)
        e = self._err(Xs)
        self.mu  = float(e.mean())
        self.sd  = float(e.std())
        self.thr = self.mu + AE_SIGMA * self.sd
        return self

    def _err(self, Xs: np.ndarray) -> np.ndarray:
        return np.mean((Xs - self.m.predict(Xs)) ** 2, axis=1)

    def score(self, X: np.ndarray):
        """Return (reconstruction_errors, anomaly_flags)."""
        Xs = self.sc.transform(X)
        e  = self._err(Xs)
        return e, e > self.thr
