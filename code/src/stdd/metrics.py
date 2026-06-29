"""Classification metrics: accuracy, calibration (ECE), rare-state recall, summaries."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error in [0, 1].

    Bins predictions by confidence and measures the gap between mean confidence
    and empirical accuracy in each bin, weighted by bin population. A
    well-calibrated classifier has ECE -> 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)
    n = y_true.size
    if n == 0:
        return 0.0
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi) if hi < 1.0 else \
                 (confidence > lo) & (confidence <= hi + 1e-12)
        m = int(in_bin.sum())
        if m == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confidence[in_bin].mean()
        ece += (m / n) * abs(acc_bin - conf_bin)
    return float(np.clip(ece, 0.0, 1.0))


def rare_state_recall(y_true: np.ndarray, y_pred: np.ndarray, rare_regime: int) -> float:
    """Recall on the rare critical regime in [0, 1]: of the true rare states, how
    many we find.

    Returns 1.0 by convention when the test set contains no rare states (nothing
    to miss), so it never spuriously penalizes a split.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos = y_true == rare_regime
    n_pos = int(pos.sum())
    if n_pos == 0:
        return 1.0
    return float(np.sum((y_pred == rare_regime) & pos) / n_pos)


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6), "ci95": round(ci95, 6), "n": int(arr.size)}
