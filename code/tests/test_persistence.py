"""Degree-0 persistent homology: the barcode is exact and its Betti-0(scale)
curve is monotone non-increasing -- a genuine multiscale topological invariant."""
import numpy as np

from stdd.persistence import h0_barcode, per_cell_isolation, persistence_features
from stdd.synthetic import make_phase_sweep


def test_barcode_has_n_minus_one_finite_bars():
    lin = make_phase_sweep(n_states=400, seed=0)
    bc = h0_barcode(lin.X, k=15)
    # a connected MST on n points has exactly n - 1 edges = n - 1 finite H0 bars.
    assert bc.n_bars == lin.n_states - 1


def test_betti0_curve_monotone_nonincreasing_in_scale():
    lin = make_phase_sweep(n_states=400, seed=1)
    bc = h0_barcode(lin.X, k=15)
    scales = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    curve = bc.betti0_curve(scales)
    for i in range(len(curve) - 1):
        assert curve[i + 1] <= curve[i], (scales[i], curve)
    # at scale 0 every state is its own component; at a large scale it collapses.
    assert bc.betti0(-1.0) == lin.n_states
    assert bc.betti0(1e6) == 1


def test_persistence_features_nonnegative():
    lin = make_phase_sweep(n_states=300, seed=2)
    bc = h0_barcode(lin.X, k=15)
    feats = persistence_features(bc)
    assert feats["max_persistence"] >= 0.0
    assert feats["total_persistence"] >= feats["max_persistence"]
    assert feats["persistence_entropy"] >= 0.0
    assert feats["n_significant"] >= 0


def test_isolation_is_per_cell_and_positive():
    lin = make_phase_sweep(n_states=300, seed=3)
    iso = per_cell_isolation(lin.X, k=15)
    assert iso.shape == (lin.n_states,)
    assert np.all(iso > 0.0)
