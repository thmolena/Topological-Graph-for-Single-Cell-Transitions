"""Forecasting + active sampling on the cell graph.

Two contributions live here, both compared against an honest baseline:

1. **Graph-smoothed forecasting** (``LabelPropagationForecaster``). The next
   cell state / branch is predicted by propagating known labels over the kNN
   graph, so a cell's forecast borrows strength from its neighbourhood on the
   data manifold. The baseline (``KNNForecaster``) is the *same neighbourhood
   information without the graph* -- a plain kNN classifier in feature space.
   Comparing the two isolates the value of the **graph diffusion**, not just of
   having neighbours.

2. **Active sampling** (``active_sample``). Given a labelling budget, choose
   which unlabeled cells to query. The contribution scores cells by
   *uncertainty x inverse local density* -- uncertainty finds the decision
   boundary, inverse density (a topology signal, see ``graph.local_density``)
   biases toward the sparse pockets where rare states hide. The baseline is
   uniform **random** sampling. We measure the lift in **rare-state recall**.

Label propagation here is the classic Zhou et al. (2004) normalized-graph
diffusion, solved by a fixed number of power iterations -- deterministic, CPU,
and a faithful surrogate for a message-passing GNN (drop in torch-geometric for
a learned propagator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import KNeighborsClassifier

from .graph import CellGraph, build_knn_graph, local_density


# --------------------------------------------------------------------------- #
# Forecasters
# --------------------------------------------------------------------------- #
@dataclass
class Forecast:
    pred: np.ndarray           # (n_test,) predicted state labels
    proba: np.ndarray          # (n_test, n_classes) class probabilities
    confidence: np.ndarray     # (n_test,) max class probability (for ECE)
    classes: np.ndarray        # (n_classes,) label for each proba column


def _onehot(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((labels.size, classes.size))
    for i, c in enumerate(labels):
        Y[i, idx[int(c)]] = 1.0
    return Y


class LabelPropagationForecaster:
    """Graph-smoothed forecaster via normalized label propagation (the method).

    Fit on the *training* cells of a split; predicts the held-out cells by
    diffusing the training labels across the full kNN graph (built once on all
    cells). ``alpha`` is the clamping / diffusion trade-off, ``n_iter`` the
    number of power iterations.
    """

    name = "graph_smoothed"

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

        # Symmetric normalization S = D^{-1/2} A D^{-1/2}.
        deg = np.asarray(A.sum(axis=1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        S = diags(dinv) @ A @ diags(dinv)

        # Seed labels: training cells clamp to their one-hot, others start at 0.
        F0 = np.zeros((X.shape[0], classes.size))
        F0[train_mask] = _onehot(y[train_mask], classes)
        F = F0.copy()
        for _ in range(self.n_iter):                 # F <- alpha S F + (1-alpha) F0
            F = self.alpha * (S @ F) + (1.0 - self.alpha) * F0

        row = F[test_mask]
        s = row.sum(axis=1, keepdims=True)
        proba = np.divide(row, s, out=np.full_like(row, 1.0 / classes.size), where=s > 0)
        pred = classes[np.argmax(proba, axis=1)]
        confidence = proba.max(axis=1)
        return Forecast(pred=pred, proba=proba, confidence=confidence, classes=classes)


class KNNForecaster:
    """Non-graph baseline: a plain kNN classifier in feature space.

    Same neighbourhood radius as the graph method but *no diffusion* -- the
    apples-to-apples control that isolates the value of label propagation.
    """

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


# --------------------------------------------------------------------------- #
# Active sampling
# --------------------------------------------------------------------------- #
def _entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def active_sample(
    X: np.ndarray,
    y: np.ndarray,
    labeled_mask: np.ndarray,
    budget: int,
    k: int = 15,
    strategy: str = "topology",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick ``budget`` unlabeled cells to query next.

    ``strategy='topology'`` (the contribution) scores each unlabeled cell by
    forecast **uncertainty** (entropy of a label-propagation forecast from the
    currently labeled set) times **inverse local density** (a topology signal
    that points at sparse, rare-state pockets), and greedily takes the top
    ``budget``. ``strategy='random'`` is the uniform baseline.

    Returns the indices (into the cell array) of the queried cells.
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

    # Uncertainty: entropy of a propagation forecast from the current labels.
    fc = LabelPropagationForecaster(k=k).predict(
        X, y, train_mask=labeled_mask, test_mask=~labeled_mask
    )
    uncertainty = _entropy(fc.proba)

    if strategy == "topology":
        # Inverse local density -> high in sparse/rare regions.
        cg = build_knn_graph(X, k=k)
        topo_signal = 1.0 / (local_density(cg, X)[unlabeled] + 1e-9)
    else:  # 'persistence': per-cell topological isolation (k-th NN distance),
        # the per-cell witness of a long degree-0 persistence bar.
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
    queried: np.ndarray, rare_state: int, k: int = 15,
) -> Tuple[float, float]:
    """Rare-state recall on the *remaining* unlabeled cells, before vs after
    adding ``queried`` to the labeled set. The gap is the active-sampling lift.
    """
    from .metrics import rare_state_recall

    def _recall(lab_mask):
        test_mask = ~lab_mask
        if test_mask.sum() == 0 or lab_mask.sum() == 0:
            return 1.0
        fc = LabelPropagationForecaster(k=k).predict(X, y, lab_mask, test_mask)
        return rare_state_recall(y[test_mask], fc.pred, rare_state)

    before = _recall(labeled_mask)
    after_mask = labeled_mask.copy()
    after_mask[queried] = True
    after = _recall(after_mask)
    return before, after
