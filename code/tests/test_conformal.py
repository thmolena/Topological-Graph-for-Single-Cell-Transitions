"""Split-conformal calibration must attain its coverage guarantee on
exchangeable data and temperature scaling must not change the predictions."""
import numpy as np

from stdd.conformal import (
    apply_temperature,
    conformal_threshold,
    empirical_coverage,
    fit_temperature,
    prediction_sets,
)


def _synthetic_scores(n, n_classes, rng):
    """Random exchangeable (probability, label) pairs."""
    logits = rng.normal(size=(n, n_classes))
    e = np.exp(logits - logits.max(1, keepdims=True))
    proba = e / e.sum(1, keepdims=True)
    y = np.array([rng.choice(n_classes, p=p) for p in proba])
    return proba, y


def test_conformal_marginal_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(0)
    delta = 0.1
    covs = []
    for _ in range(40):
        proba, y = _synthetic_scores(400, 4, rng)
        cal, test = slice(0, 200), slice(200, 400)
        thr = conformal_threshold(proba[cal], y[cal], delta=delta)
        sets = prediction_sets(proba[test], thr)
        cov = empirical_coverage(sets, y[test], rare_col=None)
        covs.append(cov.coverage)
    # average coverage is at or above the nominal 1 - delta level.
    assert np.mean(covs) >= 1 - delta - 0.02


def test_temperature_scaling_preserves_argmax():
    rng = np.random.default_rng(1)
    proba, y = _synthetic_scores(200, 4, rng)
    T = fit_temperature(proba, y)
    scaled = apply_temperature(proba, T)
    assert np.array_equal(np.argmax(proba, 1), np.argmax(scaled, 1))
    assert np.allclose(scaled.sum(1), 1.0)


def test_prediction_sets_are_nonempty_supersets_of_argmax():
    rng = np.random.default_rng(2)
    proba, y = _synthetic_scores(300, 5, rng)
    thr = conformal_threshold(proba[:150], y[:150], delta=0.1)
    sets = prediction_sets(proba[150:], thr)
    # the highest-probability class is always retained.
    assert np.all(sets[np.arange(sets.shape[0]), np.argmax(proba[150:], 1)])
