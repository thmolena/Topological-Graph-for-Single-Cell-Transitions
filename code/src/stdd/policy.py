r"""Baselines and active sampling on the state graph.

The baselines against which the spectral-truncated directed operator
(:mod:`stdd.operators`) is measured:

1. **Graph-smoothed classification** (``LabelPropagationForecaster``). Phase
   labels are propagated over the symmetric kNN graph (the classic Zhou et al.
   2004 normalized diffusion). The kNN baseline (``KNNForecaster``) is the *same
   neighbourhood information without the graph* -- a plain kNN classifier in
   feature space.

2. **Active sampling** (``active_sample``). Given a labelling budget, choose
   which unlabeled states to query. The ``persistence``/``topology`` policies
   score states by *uncertainty x topological isolation/inverse density* to bias
   toward the sparse pockets where rare critical regimes hide. The baseline is
   uniform **random** sampling, measured by rare-state recall.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import KNeighborsClassifier

from .graph import StateGraph, build_knn_graph, local_density


@dataclass
class Forecast:
    pred: np.ndarray           # (n_test,) predicted phase labels
    proba: np.ndarray          # (n_test, n_phases) phase probabilities
    confidence: np.ndarray     # (n_test,) max phase probability (for ECE)
    classes: np.ndarray        # (n_phases,) label for each proba column


def _onehot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((labels.size, classes.size))
    for i, c in enumerate(labels):
        Y[i, idx[int(c)]] = 1.0
    return Y


class LabelPropagationForecaster:
    """Graph-smoothed classifier via normalized label propagation (a baseline)."""

    name = "label_prop"

    def __init__(self, k: int = 15, alpha: float = 0.9, n_iter: int = 30):
        self.k = k
        self.alpha = alpha
        self.n_iter = n_iter

    def predict(
        self, X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
    ) -> Forecast:
        classes = np.unique(y[train_mask])
        cg = build_knn_graph(X, k=self.k)
        A = cg.adjacency.astype(float)

        deg = np.asarray(A.sum(axis=1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        S = diags(dinv) @ A @ diags(dinv)

        F0 = np.zeros((X.shape[0], classes.size))
        F0[train_mask] = _onehot(y[train_mask], classes)
        F = F0.copy()
        for _ in range(self.n_iter):
            F = self.alpha * (S @ F) + (1.0 - self.alpha) * F0

        row = F[test_mask]
        s = row.sum(axis=1, keepdims=True)
        proba = np.divide(row, s, out=np.full_like(row, 1.0 / classes.size), where=s > 0)
        pred = classes[np.argmax(proba, axis=1)]
        confidence = proba.max(axis=1)
        return Forecast(pred=pred, proba=proba, confidence=confidence, classes=classes)


class KNNForecaster:
    """Non-graph baseline: a plain (inductive) kNN classifier in feature space."""

    name = "baseline_knn"

    def __init__(self, k: int = 15):
        self.k = k

    def predict(
        self, X: np.ndarray, y: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray
    ) -> Forecast:
        k = int(np.clip(self.k, 1, max(1, int(train_mask.sum()) - 1)))
        clf = KNeighborsClassifier(n_neighbors=k).fit(X[train_mask], y[train_mask])
        proba = clf.predict_proba(X[test_mask])
        classes = clf.classes_
        pred = classes[np.argmax(proba, axis=1)]
        confidence = proba.max(axis=1)
        return Forecast(pred=pred, proba=proba, confidence=confidence, classes=classes)


def _entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def active_sample(
    X: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    budget: int,
    k: int = 15,
    strategy: str = "persistence",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick ``budget`` unlabeled states to query next.

    ``strategy='persistence'`` scores each unlabeled state by classifier
    **uncertainty** (entropy of a label-propagation forecast) times per-state
    **topological isolation** (the persistence witness); ``'topology'`` uses
    inverse local density instead; ``'random'`` is the uniform baseline.
    """
    rng = rng or np.random.default_rng(0)
    unlabeled = np.flatnonzero(~labeled_mask)
    if unlabeled.size == 0:
        return np.array([], dtype=int)
    budget = int(min(budget, unlabeled.size))

    if strategy == "random":
        return rng.choice(unlabeled, size=budget, replace=False)

    if strategy not in ("topology", "persistence"):
        raise ValueError(f"unknown active-sampling strategy {strategy!r}")

    fc = LabelPropagationForecaster(k=k).predict(
        X, y, train_mask=labeled_mask, test_mask=~labeled_mask
    )
    uncertainty = _entropy(fc.proba)

    if strategy == "topology":
        cg = build_knn_graph(X, k=k)
        topo_signal = 1.0 / (local_density(cg, X)[unlabeled] + 1e-9)
    else:  # 'persistence': per-state topological isolation (k-th NN distance),
        # the per-state witness of a long degree-0 persistence bar.
        from .persistence import per_cell_isolation
        topo_signal = per_cell_isolation(X, k=k)[unlabeled]

    score = _zscore(uncertainty) + _zscore(topo_signal)
    order = np.argsort(-score)
    return unlabeled[order[:budget]]


def _zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    sd = v.std()
    return (v - v.mean()) / (sd + 1e-9)


def recall_after_query(
    X: np.ndarray, y: np.ndarray, labeled_mask: np.ndarray,
    queried: np.ndarray, rare_regime: int, k: int = 15,
) -> Tuple[float, float]:
    """Rare-state recall on the *remaining* unlabeled states, before vs after
    adding ``queried`` to the labeled set. The gap is the active-sampling lift.
    """
    from .metrics import rare_state_recall

    def _recall(lab_mask):
        test_mask = ~lab_mask
        if test_mask.sum() == 0 or lab_mask.sum() == 0:
            return 1.0
        fc = LabelPropagationForecaster(k=k).predict(X, y, lab_mask, test_mask)
        return rare_state_recall(y[test_mask], fc.pred, rare_regime)

    before = _recall(labeled_mask)
    after_mask = labeled_mask.copy()
    after_mask[queried] = True
    after = _recall(after_mask)
    return before, after
