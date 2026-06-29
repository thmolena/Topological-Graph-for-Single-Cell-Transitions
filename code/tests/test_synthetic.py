"""The synthetic lineage must have the structure the benchmark advertises."""
import numpy as np

from stdd.synthetic import make_lineage


def test_branches_and_states():
    lin = make_lineage(n_cells=600, n_branches=3, rare_fraction=0.05, seed=0)
    # exactly 3 branches, plus one carved-out rare state => 4 state labels.
    assert len(np.unique(lin.branch)) == 3
    assert lin.n_states == 4
    assert lin.rare_state == 3


def test_rare_state_is_rare():
    lin = make_lineage(n_cells=600, rare_fraction=0.05, seed=1)
    freqs = lin.state_frequencies()
    rare_freq = freqs[lin.rare_state]
    # rare but present: a small minority of cells, never empty.
    assert 0.0 < rare_freq < 0.12
    # the rare state must be the least frequent state.
    assert rare_freq == min(freqs.values())


def test_pseudotime_ordering_sane():
    lin = make_lineage(n_cells=600, seed=2)
    # pseudotime is a valid progression coordinate in [0, 1].
    assert lin.pseudotime.min() >= 0.0
    assert lin.pseudotime.max() <= 1.0
    # fate branches only exist after the split point (they have no early cells).
    fate = lin.branch == 2
    assert lin.pseudotime[fate].min() >= 0.45 - 1e-9
    # the trunk spans earlier pseudotime than the fate tips on average.
    assert lin.pseudotime[lin.branch == 0].mean() < lin.pseudotime[fate].mean()


def test_seeded_reproducible():
    a = make_lineage(n_cells=300, seed=7)
    b = make_lineage(n_cells=300, seed=7)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.state, b.state)
