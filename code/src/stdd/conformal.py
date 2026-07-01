r"""Split-conformal calibration of the diffused phase classifier.

Graph smoothing buys accuracy and rare-state recall at the price of a calibration
penalty. This module supplies the repair, with two tools that both consume a
held-out *calibration* slice of the annotated states.

1. **Temperature scaling** (:func:`fit_temperature`). A scalar ``T`` is fit by
   minimizing calibration negative log-likelihood; rescaling the diffused logits
   by ``T`` moves the confidences toward the empirical accuracy and reduces ECE
   without changing the predicted phase.

2. **Split-conformal prediction sets** (:func:`conformal_threshold`,
   :func:`prediction_sets`). Using the nonconformity score
   :math:`s = 1 - \hat p(\text{true phase})` on the calibration states, the
   conformal quantile is a threshold that yields prediction sets with finite-
   sample marginal coverage :math:`\ge 1-\delta` under exchangeability of
   calibration and test states.

Pure numpy; deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logits(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-9, 1.0))


def fit_temperature(proba_cal: np.ndarray, y_cal_idx: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Fit a temperature ``T`` by grid-minimizing calibration NLL."""
    if proba_cal.shape[0] == 0:
        return 1.0
    grid = grid if grid is not None else np.linspace(0.25, 6.0, 116)
    logits = _logits(proba_cal)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        z = logits / T
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -logp[np.arange(logits.shape[0]), y_cal_idx].mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Re-normalize probabilities under temperature ``T`` (phase-preserving)."""
    z = _logits(proba) / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def conformal_threshold(proba_cal: np.ndarray, y_cal_idx: np.ndarray,
                        delta: float = 0.1) -> float:
    """Split-conformal score threshold for marginal coverage ``>= 1 - delta``."""
    n = proba_cal.shape[0]
    if n == 0:
        return 1.0
    scores = 1.0 - proba_cal[np.arange(n), y_cal_idx]
    level = min(1.0, np.ceil((n + 1) * (1.0 - delta)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean ``(n_test, n_phases)`` membership: phase kept iff ``1 - p <= thr``."""
    return (1.0 - proba) <= threshold + 1e-12


@dataclass
class CoverageReport:
    coverage: float        # fraction of test states whose true phase is in the set
    avg_set_size: float    # mean number of phases per prediction set
    rare_coverage: float   # coverage restricted to true rare states (1.0 if none)


def empirical_coverage(sets: np.ndarray, y_test_idx: np.ndarray,
                       rare_col: int | None) -> CoverageReport:
    """Realized marginal (and rare-conditional) coverage of conformal sets."""
    n = sets.shape[0]
    if n == 0:
        return CoverageReport(1.0, 0.0, 1.0)
    covered = sets[np.arange(n), y_test_idx]
    cov = float(covered.mean())
    size = float(sets.sum(axis=1).mean())
    if rare_col is None:
        rare_cov = 1.0
    else:
        is_rare = y_test_idx == rare_col
        rare_cov = float(covered[is_rare].mean()) if is_rare.any() else 1.0
    return CoverageReport(coverage=cov, avg_set_size=size, rare_coverage=rare_cov)
