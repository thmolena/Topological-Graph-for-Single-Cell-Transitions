r"""Spectral-truncated, drive-directed (noncommutative) graph operators.

This module is the methodological core of the package. It replaces the plain,
symmetric label-propagation diffusion of :mod:`stdd.policy` with a family of
operators built from two ingredients that the symmetric smoother lacks:

1. **A drive-directed propagator** ``P``. The standard graph smoother diffuses
   labels through the *self-adjoint* normalized adjacency
   :math:`S=D^{-1/2}AD^{-1/2}`, which is blind to the arrow of the control-
   parameter sweep. We instead reweight each directed edge :math:`i\to j` by a
   bounded forward kernel :math:`\kappa(t_j-t_i)=\exp\!\big(\beta\tanh((t_j-t_i)/\tau)\big)`
   of the control increment and row-normalize, giving a *non-self-adjoint*
   (``P P^T != P^T P``) Markov operator that transports labels along the sweep.
   Non-normality is the state-graph instance of the operator *noncommutativity*
   studied in C*-algebraic kernel machines: the commutator :math:`[P,P^\top]` is
   non-zero exactly when the control gradient varies across edges
   (:func:`nonnormality`).

2. **Spectral truncation.** We project the diffusion onto the span of the leading
   ``rank`` graph-Fourier modes (the top eigenvectors of ``S``). The truncated
   propagator :math:`B=\Pi_r P\,\Pi_r` is a band-limited, denoised surrogate
   solved in closed form in :math:`r` coordinates. Truncation is the graph
   analogue of the spectral truncation of multiplication operators in
   noncommutative geometry: the truncated product is *not* the product of the
   truncations (:func:`truncation_commutator_norm`), and the truncated
   multiplications generate a noncommutative operator ``*``-algebra with a
   positive-definite Gram kernel (:func:`truncated_operator_kernel`).

Two classifiers are exposed: ``SpectralTruncatedDiffusion`` (transductive) and
``InductiveSpectralTruncatedDiffusion`` (inductive, via an out-of-sample Nystrom
extension that never sees held-out connectivity). Everything is CPU
numpy/scipy/scikit-learn and fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from .graph import build_knn_graph
from .policy import Forecast


# --------------------------------------------------------------------------- #
# Graph-Fourier basis (spectral truncation)
# --------------------------------------------------------------------------- #
@dataclass
class SpectralBasis:
    """Leading ``rank`` graph-Fourier modes of the symmetric normalized graph."""

    U: np.ndarray          # (n_states, rank) orthonormal low-frequency modes
    vals: np.ndarray       # (rank,) eigenvalues of S, descending
    deg: np.ndarray        # (n_states,) node degrees
    dinv: np.ndarray       # (n_states,) D^{-1/2}

    @property
    def rank(self) -> int:
        return self.U.shape[1]


def graph_fourier_basis(A, rank: int) -> SpectralBasis:
    """Leading ``rank`` graph-Fourier modes of a symmetric graph ``A``.

    Returns the eigenvectors of ``S = D^{-1/2} A D^{-1/2}`` with the largest
    eigenvalues, the smoothest functions on the graph, spanning the band-limited
    subspace used by the truncated diffusion. A fixed ARPACK starting vector makes
    the decomposition deterministic; a dense solve is used on small graphs.
    """
    A = A.astype(float)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    S = diags(dinv) @ A @ diags(dinv)
    rank = int(min(rank, n - 2))
    rank = max(rank, 1)
    if n <= 600 or rank >= n - 2:
        w, V = np.linalg.eigh(S.toarray())
        idx = np.argsort(-w)[:rank]
        vals, U = w[idx], V[:, idx]
    else:
        from scipy.sparse.linalg import ArpackNoConvergence
        ncv = int(min(n - 1, max(2 * rank + 1, 40)))
        v0 = np.random.default_rng(0).standard_normal(n)
        try:
            vals, U = eigsh(S.tocsc(), k=rank, which="LA", ncv=ncv,
                            maxiter=20000, tol=0.0, v0=v0)
        except ArpackNoConvergence as e:  # pragma: no cover - machine dependent
            vals, U = e.eigenvalues, e.eigenvectors
            if vals.size < rank:
                w, V = np.linalg.eigh(S.toarray())
                idx = np.argsort(-w)[:rank]
                vals, U = w[idx], V[:, idx]
    order = np.argsort(-vals)
    return SpectralBasis(U=U[:, order], vals=vals[order], deg=deg, dinv=dinv)


# --------------------------------------------------------------------------- #
# Drive-directed (noncommutative) propagator
# --------------------------------------------------------------------------- #
def directed_propagator(A, control: np.ndarray, beta: float, tau: float = 0.2):
    """Row-stochastic, drive-directed transition operator on the state graph.

    Each directed edge ``i -> j`` of the symmetric graph ``A`` is reweighted by
    the bounded forward kernel ``kappa(dt) = exp(beta * tanh(dt / tau))`` of the
    control increment ``dt = t_j - t_i`` (>1 forward, <1 backward), then each row
    is normalized. ``beta = 0`` recovers the undirected random walk. The result is
    non-self-adjoint -- the noncommutative operator at the heart of the method.
    """
    A = A.astype(float).tocoo()
    dt = control[A.col] - control[A.row]
    w = A.data * np.exp(beta * np.tanh(dt / max(tau, 1e-6)))
    W = csr_matrix((w, (A.row, A.col)), shape=A.shape)
    dW = np.asarray(W.sum(axis=1)).ravel()
    return diags(1.0 / np.maximum(dW, 1e-12)) @ W


def nonnormality(P) -> float:
    """Relative non-normality ``||P P^T - P^T P||_F / ||P||_F^2`` of ``P``.

    Zero iff ``P`` is normal (e.g. a regular undirected walk); strictly positive
    once the control direction makes the operator non-self-adjoint.
    """
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    comm = P @ P.T - P.T @ P
    denom = (np.linalg.norm(P, "fro") ** 2) + 1e-12
    return float(np.linalg.norm(comm, "fro") / denom)


def truncated_directed_operator(basis: SpectralBasis, P, eps: float) -> np.ndarray:
    """Band-limited, lazy directed transport ``B = (1-eps) I_r + eps U^T P U``."""
    U = basis.U
    B = U.T @ (P @ U)
    r = B.shape[0]
    return (1.0 - eps) * np.eye(r) + eps * B


# --------------------------------------------------------------------------- #
# Reduced-coordinate diffusion solve (closed form)
# --------------------------------------------------------------------------- #
def _onehot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((labels.size, classes.size))
    for i, c in enumerate(labels):
        Y[i, idx[int(c)]] = 1.0
    return Y


def _band_limited_field(basis: SpectralBasis, B: np.ndarray, F0: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Closed-form truncated directed diffusion ``F = U (I-alpha B)^{-1}(1-alpha) U^T F0``."""
    U = basis.U
    r = B.shape[0]
    g0 = U.T @ F0
    C = np.linalg.solve(np.eye(r) - alpha * B, (1.0 - alpha) * g0)
    return U @ C


def _field_to_proba(field_rows: np.ndarray, n_classes: int):
    pos = np.maximum(field_rows, 0.0)
    s = pos.sum(axis=1, keepdims=True)
    proba = np.divide(pos, s, out=np.full_like(pos, 1.0 / n_classes), where=s > 0)
    return proba, field_rows


# --------------------------------------------------------------------------- #
# Transductive classifier
# --------------------------------------------------------------------------- #
class SpectralTruncatedDiffusion:
    """Transductive spectral-truncated directed diffusion classifier."""

    name = "std_transductive"

    def __init__(self, k: int = 20, rank: int = 80, alpha: float = 0.9,
                 beta: float = 1.0, eps: float = 0.6, tau: float = 0.2):
        self.k = k
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tau = tau

    def predict(self, X, y, train_mask, test_mask, control) -> Forecast:
        classes = np.unique(y[train_mask])
        cg = build_knn_graph(X, k=self.k)
        A = cg.adjacency.astype(float)
        basis = graph_fourier_basis(A, self.rank)
        P = directed_propagator(A, control, self.beta, self.tau)
        B = truncated_directed_operator(basis, P, self.eps)
        F0 = np.zeros((X.shape[0], classes.size))
        F0[train_mask] = _onehot(y[train_mask], classes)
        F = _band_limited_field(basis, B, F0, self.alpha)
        proba, pred_field = _field_to_proba(F[test_mask], classes.size)
        pred = classes[np.argmax(pred_field, axis=1)]
        return Forecast(pred=pred, proba=proba, confidence=proba.max(axis=1), classes=classes)


# --------------------------------------------------------------------------- #
# Inductive classifier (out-of-sample Nystrom extension)
# --------------------------------------------------------------------------- #
class InductiveSpectralTruncatedDiffusion:
    """Inductive spectral-truncated directed diffusion classifier.

    The graph, the graph-Fourier basis and the diffused field are built on the
    *training* states only. Held-out states are labelled by an out-of-sample
    Nystrom extension: each test state borrows the band-limited field of its
    nearest *training* states through a Gaussian-weighted average, so no
    test--test edge and no held-out connectivity is used at inference. This is the
    matched inductive control, and it transfers to shadow data from an unseen
    device or schedule without rebuilding the graph.
    """

    name = "std_inductive"

    def __init__(self, k: int = 20, rank: int = 80, alpha: float = 0.9,
                 beta: float = 1.0, eps: float = 0.6, tau: float = 0.2):
        self.k = k
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.tau = tau

    def _train_field(self, X_tr, t_tr, y, train_idx, labeled_local, classes):
        cg = build_knn_graph(X_tr, k=self.k)
        A = cg.adjacency.astype(float)
        basis = graph_fourier_basis(A, self.rank)
        P = directed_propagator(A, t_tr, self.beta, self.tau)
        B = truncated_directed_operator(basis, P, self.eps)
        F0 = np.zeros((X_tr.shape[0], classes.size))
        F0[labeled_local] = _onehot(y[train_idx][labeled_local], classes)
        return _band_limited_field(basis, B, F0, self.alpha)

    def predict(self, X, y, train_mask, test_mask, labeled_mask, control) -> Forecast:
        classes = np.unique(y[labeled_mask])
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        X_tr, t_tr = X[train_idx], control[train_idx]
        labeled_local = np.isin(train_idx, np.flatnonzero(labeled_mask))
        F_tr = self._train_field(X_tr, t_tr, y, train_idx, labeled_local, classes)

        k = int(min(self.k, max(1, X_tr.shape[0] - 1)))
        nn = NearestNeighbors(n_neighbors=k).fit(X_tr)
        dist, idx = nn.kneighbors(X[test_idx])
        bw = np.median(dist) + 1e-9
        w = np.exp(-(dist ** 2) / (bw ** 2))
        w /= w.sum(axis=1, keepdims=True)
        F_te = np.einsum("tk,tkc->tc", w, F_tr[idx])
        proba, pred_field = _field_to_proba(F_te, classes.size)
        pred = classes[np.argmax(pred_field, axis=1)]
        return Forecast(pred=pred, proba=proba, confidence=proba.max(axis=1), classes=classes)


# --------------------------------------------------------------------------- #
# Spectral-truncation noncommutativity diagnostics (C*-algebraic flavour)
# --------------------------------------------------------------------------- #
def truncated_multiplication(basis: SpectralBasis, phi: np.ndarray) -> np.ndarray:
    """Truncated multiplication operator ``M_phi = U^T diag(phi) U`` (``r x r``)."""
    U = basis.U
    return U.T @ (phi[:, None] * U)


def truncation_commutator_norm(basis: SpectralBasis, phi: np.ndarray,
                               psi: np.ndarray) -> float:
    """``||[M_phi, M_psi]||_F`` for two truncated multiplication operators.

    Pointwise multiplication is commutative, but its spectral truncations are
    not. The Frobenius norm of the commutator is the witness of this
    noncommutativity and is zero only in the full-rank limit.
    """
    Mphi = truncated_multiplication(basis, phi)
    Mpsi = truncated_multiplication(basis, psi)
    comm = Mphi @ Mpsi - Mpsi @ Mphi
    return float(np.linalg.norm(comm, "fro"))


def truncated_operator_kernel(basis: SpectralBasis, features: np.ndarray) -> np.ndarray:
    """Positive-semidefinite Gram operator of the truncated multiplications.

    For feature channels ``phi_c`` (columns of ``features``) define
    ``K = sum_c M_{phi_c}^{(r)} (M_{phi_c}^{(r)})^*`` on the band-limited
    subspace. As a sum of ``A A^*`` terms it is positive semidefinite (and
    positive definite when ``kernel_min_eigenvalue > 0``); its generators -- the
    truncated multiplication operators -- do not commute, so ``K`` is the
    symmetric Gram object of a genuinely noncommutative truncated operator
    ``*``-algebra on the state graph.
    """
    r = basis.rank
    K = np.zeros((r, r))
    for c in range(features.shape[1]):
        M = truncated_multiplication(basis, features[:, c])
        K += M @ M.T
    return K


def kernel_min_eigenvalue(K: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix (the PSD witness, ``>= 0``)."""
    return float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())


def spectral_radius(M: np.ndarray) -> float:
    """Spectral radius ``max_i |lambda_i(M)|`` of a square matrix."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))
