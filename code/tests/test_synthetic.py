"""The synthetic lineage must have the structure the benchmark advertises."""
import numpy as np

from stdd.synthetic import make_phase_sweep


def test_branches_and_states():
    lin = make_phase_sweep(n_states=600, n_phase_branches=3, rare_fraction=0.05, seed=0)
    # exactly 3 branches, plus one carved-out rare phase => 4 phase labels.
    assert len(np.unique(lin.phase_branch)) == 3
    assert lin.n_phases == 4
    assert lin.rare_regime == 3


def test_rare_regime_is_rare():
    lin = make_phase_sweep(n_states=600, rare_fraction=0.05, seed=1)
    freqs = lin.phase_frequencies()
    rare_freq = freqs[lin.rare_regime]
    # rare but present: a small minority of states, never empty.
    assert 0.0 < rare_freq < 0.12
    # the rare phase must be the least frequent phase.
    assert rare_freq == min(freqs.values())


def test_control_ordering_sane():
    lin = make_phase_sweep(n_states=600, seed=2)
    # control is a valid progression coordinate in [0, 1].
    assert lin.control.min() >= 0.0
    assert lin.control.max() <= 1.0
    # fate branches only exist after the split point (they have no early states).
    fate = lin.phase_branch == 2
    assert lin.control[fate].min() >= 0.45 - 1e-9
    # the trunk spans earlier control than the fate tips on average.
    assert lin.control[lin.phase_branch == 0].mean() < lin.control[fate].mean()


def test_seeded_reproducible():
    a = make_phase_sweep(n_states=300, seed=7)
    b = make_phase_sweep(n_states=300, seed=7)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.phase, b.phase)
