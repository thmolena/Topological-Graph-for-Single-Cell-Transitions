"""Metric bounds: accuracy, ECE, rare-phase recall all live in [0, 1]."""
import numpy as np
import pytest

from stdd import metrics


def test_accuracy_bounds():
    y = np.array([0, 1, 2, 2, 1])
    assert metrics.accuracy(y, y) == 1.0
    assert metrics.accuracy(y, np.array([2, 2, 2, 2, 2])) == pytest.approx(0.4)
    assert 0.0 <= metrics.accuracy(y, np.zeros_like(y)) <= 1.0


def test_ece_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 50
        y_true = rng.integers(0, 3, size=n)
        y_pred = rng.integers(0, 3, size=n)
        conf = rng.uniform(0.34, 1.0, size=n)
        ece = metrics.expected_calibration_error(y_true, y_pred, conf)
        assert 0.0 <= ece <= 1.0


def test_rare_recall_bounds():
    y_true = np.array([0, 0, 3, 3, 1])
    # found both rare states -> recall 1
    assert metrics.rare_state_recall(y_true, np.array([0, 0, 3, 3, 1]), rare_regime=3) == 1.0
    # found one of two -> recall 0.5
    assert metrics.rare_state_recall(y_true, np.array([0, 0, 3, 0, 1]), rare_regime=3) == 0.5
    # no rare states in truth -> recall defined as 1 (nothing to miss)
    assert metrics.rare_state_recall(np.array([0, 1, 2]), np.array([0, 1, 2]), rare_regime=3) == 1.0
    r = metrics.rare_state_recall(y_true, np.zeros_like(y_true), rare_regime=3)
    assert 0.0 <= r <= 1.0


def test_summarize_shape():
    s = metrics.summarize([0.8, 0.9, 0.7])
    assert set(s) == {"mean", "std", "ci95", "n"}
    assert s["n"] == 3
    assert 0.0 <= s["mean"] <= 1.0
