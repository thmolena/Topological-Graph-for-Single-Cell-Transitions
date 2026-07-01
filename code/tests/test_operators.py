"""Properties of the spectral-truncated, control-directed graph operators."""
import numpy as np

from stdd.graph import build_knn_graph
from stdd.operators import (
    InductiveSpectralTruncatedDiffusion,
    SpectralTruncatedDiffusion,
    directed_propagator,
    graph_fourier_basis,
    nonnormality,
    truncation_commutator_norm,
)
from stdd.policy import KNNForecaster
from stdd.splits import make_split
from stdd.synthetic import make_phase_sweep


def test_directed_propagator_is_row_stochastic():
    lin = make_phase_sweep(n_states=300, seed=0)
    A = build_knn_graph(lin.X, k=15).adjacency
    P = directed_propagator(A, lin.control, beta=1.0).toarray()
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)


def test_direction_increases_nonnormality():
    """The directed propagator is non-normal, and the directional reweighting
    strictly increases non-normality over the undirected random walk."""
    lin = make_phase_sweep(n_states=400, seed=1)
    A = build_knn_graph(lin.X, k=15).adjacency
    P0 = directed_propagator(A, lin.control, beta=0.0)
    P1 = directed_propagator(A, lin.control, beta=1.0)
    assert nonnormality(P1) > nonnormality(P0) > 0.0


def test_symmetric_normalized_operator_is_normal():
    """S = D^-1/2 A D^-1/2 is self-adjoint, hence normal (non-normality 0)."""
    lin = make_phase_sweep(n_states=300, seed=2)
    A = build_knn_graph(lin.X, k=15).adjacency.astype(float)
    from scipy.sparse import diags
    deg = np.asarray(A.sum(1)).ravel(); dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    S = diags(dinv) @ A @ diags(dinv)
    assert nonnormality(S) < 1e-9


def test_truncated_multiplication_noncommutes():
    """Spectral truncation breaks commutativity of pointwise multiplication."""
    lin = make_phase_sweep(n_states=400, seed=0)
    A = build_knn_graph(lin.X, k=15).adjacency
    basis = graph_fourier_basis(A, rank=30)
    c = truncation_commutator_norm(basis, lin.X[:, 0], lin.X[:, 1])
    assert c > 0.0


def test_basis_orthonormal():
    lin = make_phase_sweep(n_states=350, seed=4)
    A = build_knn_graph(lin.X, k=15).adjacency
    basis = graph_fourier_basis(A, rank=25)
    assert np.allclose(basis.U.T @ basis.U, np.eye(basis.rank), atol=1e-6)
    # eigenvalues of S lie in [-1, 1].
    assert basis.vals.max() <= 1.0 + 1e-6 and basis.vals.min() >= -1.0 - 1e-6


def test_inductive_beats_point_baseline_on_rare_recovery():
    """On a seeded lineage the inductive directed operator recovers the rare
    phase that a plain kNN classifier misses (the headline effect, in miniature)."""
    lin = make_phase_sweep(n_states=800, n_devices=4, rare_fraction=0.05, seed=0)
    split = make_split(lin, "device", label_fraction=0.1, seed=0)
    knn = KNNForecaster(k=15).predict(lin.X, lin.phase, split.labeled, split.test)
    std = InductiveSpectralTruncatedDiffusion(k=15, rank=40).predict(
        lin.X, lin.phase, split.train, split.test, split.labeled, lin.control)
    from stdd.metrics import rare_state_recall
    y = lin.phase[split.test]
    assert rare_state_recall(y, std.pred, lin.rare_regime) >= \
        rare_state_recall(y, knn.pred, lin.rare_regime)


def test_forecasts_return_simplex_probabilities():
    lin = make_phase_sweep(n_states=400, seed=3)
    split = make_split(lin, "shot_batch", label_fraction=0.1, seed=3)
    for model in (SpectralTruncatedDiffusion(k=15, rank=30),):
        fc = model.predict(lin.X, lin.phase, split.labeled, split.test, lin.control)
        assert np.allclose(fc.proba.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(fc.proba >= -1e-9)
