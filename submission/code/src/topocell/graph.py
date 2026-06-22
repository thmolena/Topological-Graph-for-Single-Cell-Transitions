r"""kNN cell graph and its topology.

A single-cell dataset becomes a graph by connecting each cell to its ``k``
nearest neighbours in feature space (the same construction scanpy uses before
clustering / pseudotime). Two things matter here:

  * the graph is the substrate for **label propagation** (graph-smoothed
    forecasting), so its connectivity controls how labels spread;
  * the **number of connected components** is a discrete topological invariant
    (the rank of \(H_0\), i.e. Betti-0). As ``k`` grows the graph can only gain
    edges, so components can only *merge* -- the component count is monotone
    non-increasing in ``k``. That monotonicity is a real, testable property and
    a sanity check on the build (``tests/test_betti_monotonic.py``).

Everything here is CPU numpy/scipy/scikit-learn/networkx. A GPU build (RAPIDS
cuML kNN + cuGraph connected-components) is a drop-in for large cohorts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


@dataclass
class CellGraph:
    """A built kNN graph plus the artefacts the policy needs."""

    k: int
    knn_idx: np.ndarray        # (n_cells, k) neighbour indices (self excluded)
    adjacency: "object"        # scipy CSR sparse symmetric adjacency
    graph: nx.Graph            # networkx view (for component / topology queries)

    @property
    def n_cells(self) -> int:
        return self.adjacency.shape[0]


def build_knn_graph(X: np.ndarray, k: int = 15) -> CellGraph:
    """Build a symmetric kNN graph on cell features.

    ``k`` is clamped to ``n_cells - 1``. The adjacency is symmetrized (a mutual-
    or-either-direction kNN), which is the standard single-cell convention and
    keeps connected-components well defined.
    """
    n = X.shape[0]
    k = int(np.clip(k, 1, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)       # +1: self is neighbour 0
    _, idx = nn.kneighbors(X)
    knn_idx = idx[:, 1:]                                   # drop self column

    # Build a symmetric sparse adjacency from the directed kNN edges.
    rows = np.repeat(np.arange(n), k)
    cols = knn_idx.reshape(-1)
    from scipy.sparse import csr_matrix

    data = np.ones(rows.size)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)                                     # symmetrize (either-direction)
    A.setdiag(0)
    A.eliminate_zeros()

    g = nx.from_scipy_sparse_array(A)
    return CellGraph(k=k, knn_idx=knn_idx, adjacency=A, graph=g)


def n_connected_components(cg: CellGraph) -> int:
    r"""Number of connected components = Betti-0 (rank of \(H_0\))."""
    n_comp, _ = connected_components(cg.adjacency, directed=False)
    return int(n_comp)


def betti0_curve(X: np.ndarray, ks: List[int]) -> List[int]:
    """Component count as a function of ``k`` -- monotone non-increasing.

    This is the topological feature the method conditions on: a cell population
    that stays fragmented as ``k`` grows is genuinely multi-modal (several
    states), whereas one that collapses to a single component at small ``k`` is
    a continuum. The curve is returned in the order of ``ks``.
    """
    counts = []
    for k in sorted(ks):
        cg = build_knn_graph(X, k=k)
        counts.append(n_connected_components(cg))
    return counts


def topology_features(cg: CellGraph) -> dict:
    """Cheap graph-level topology descriptors used by the active-sampling policy.

    All are graph invariants (no node ordering): component count (Betti-0),
    mean degree, degree variance, and the mean local clustering coefficient
    (a triangle / cycle proxy that correlates with manifold curvature).
    """
    g = cg.graph
    degs = np.array([d for _, d in g.degree()], dtype=float)
    return {
        "betti0": n_connected_components(cg),
        "mean_degree": float(degs.mean()) if degs.size else 0.0,
        "degree_var": float(degs.var()) if degs.size else 0.0,
        "mean_clustering": float(nx.average_clustering(g)) if g.number_of_nodes() else 0.0,
    }


def local_density(cg: CellGraph, X: np.ndarray) -> np.ndarray:
    """Per-cell inverse mean kNN distance -- high in dense regions, low in rare ones.

    Used by the active-sampling policy: rare states sit in *low-density* pockets,
    so 1/density is a cheap, topology-aware proxy for "where the rare cells are".
    """
    n = X.shape[0]
    d = np.zeros(n)
    for i in range(n):
        nb = cg.knn_idx[i]
        d[i] = np.linalg.norm(X[nb] - X[i], axis=1).mean()
    return 1.0 / (d + 1e-9)
