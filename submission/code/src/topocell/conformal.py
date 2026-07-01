r"""Split-conformal calibration of the diffused forecast.

The symmetric-baseline study reported that graph smoothing buys accuracy and
rare-state recall at the price of a small but consistent *calibration* penalty
(worse expected calibration error on every split), and left recalibration as
future work. This module supplies it, with two complementary tools that both
consume a held-out *calibration* slice of the annotated cells.

1. **Temperature scaling** (:func:`fit_temperature`). A single scalar ``T`` is
   fit by minimizing calibration negative log-likelihood; dividing the diffused
   logits by ``T`` rescales the confidences toward the empirical accuracy and
   reduces expected calibration error without changing the predicted class.

2. **Split-conformal prediction sets** (:func:`conformal_threshold`,
   :func:`prediction_sets`). Using the nonconformity score
   :math:`s = 1 - \hat p(\text{true class})` on the calibration cells, the
   :math:`(1-\delta)(1+1/n_\mathrm{cal})` empirical quantile is a threshold that
   yields prediction sets with finite-sample marginal coverage
   :math:`\ge 1-\delta` under exchangeability of calibration and test cells
   (:func:`empirical_coverage` measures the realized coverage, including under
   the transfer splits where exchangeability is only approximate). The rare
   state is *retained* in the set whenever its diffused probability is within the
   threshold, giving a distribution-free handle on rare-state inclusion.

Pure numpy; deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logits(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, 1e-9, 1.0))


def fit_temperature(proba_cal: np.ndarray, y_cal_idx: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Fit a temperature ``T`` by grid-minimizing calibration NLL.

    ``proba_cal`` are predicted class probabilities on the calibration cells and
    ``y_cal_idx`` the index of the true class (into the column order). Returns
    ``T >= 1`` softening (or ``< 1`` sharpening) the confidences. A grid search
    is used for determinism and to avoid an optimizer dependency.
    """
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
    """Re-normalize probabilities under temperature ``T`` (class-preserving)."""
    z = _logits(proba) / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def conformal_threshold(proba_cal: np.ndarray, y_cal_idx: np.ndarray,
                        delta: float = 0.1) -> float:
    """Split-conformal score threshold for marginal coverage ``>= 1 - delta``.

    The nonconformity scores are ``s_i = 1 - p_i(true class)``; the threshold is
    the ``ceil((n+1)(1-delta)) / n`` empirical quantile of the calibration
    scores (the finite-sample-valid conformal level).
    """
    n = proba_cal.shape[0]
    if n == 0:
        return 1.0
    scores = 1.0 - proba_cal[np.arange(n), y_cal_idx]
    level = min(1.0, np.ceil((n + 1) * (1.0 - delta)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean ``(n_test, n_classes)`` membership: class kept iff ``1 - p <= thr``."""
    return (1.0 - proba) <= threshold + 1e-12


@dataclass
class CoverageReport:
    coverage: float        # fraction of test cells whose true class is in the set
    avg_set_size: float    # mean number of classes per prediction set
    rare_coverage: float   # coverage restricted to true rare cells (1.0 if none)


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
