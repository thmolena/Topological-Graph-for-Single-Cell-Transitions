r"""Persistent homology of the cell point cloud (degree-0 barcode).

The symmetric-baseline study used a single, fixed-scale topological number --
the connected-component count (Betti-0) of one ``k``NN graph -- which at the
reported scale is the constant ``1`` and therefore carries no information. Here
we replace it with a genuine *multiscale* topological descriptor: the degree-0
persistence barcode of the cell point cloud over the full Vietoris--Rips-style
distance filtration.

For :math:`H_0`, persistence is computed *exactly* and cheaply by union--find:
as the connectivity scale :math:`\epsilon` grows from :math:`0`, every cell is
born at :math:`\epsilon = 0` as its own component, and two components merge at
the length of the edge that first joins them. The multiset of merge lengths is
exactly the multiset of edge weights of a Euclidean minimum spanning tree
(Kruskal's algorithm), so the finite :math:`H_0` bars are the MST edge weights
and ``betti0(eps) = #{bars > eps} + 1``. Long bars correspond to well-separated
sub-populations -- a deliberately rare, spatially isolated state produces a
long-lived bar -- so the barcode is an informative, scale-aware topology signal
rather than a passive sanity check.

Pure numpy/scipy; no external topology dependency (``ripser``/``gudhi``) is
required for the degree-0 invariant used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors


@dataclass
class Barcode:
    """Degree-0 persistence barcode (finite bars; all births at 0)."""

    deaths: np.ndarray     # (n_cells - 1,) sorted finite death scales = MST edge weights

    @property
    def n_bars(self) -> int:
        return int(self.deaths.size)

    def betti0(self, eps: float) -> int:
        """Number of connected components at connectivity scale ``eps``."""
        return int(np.sum(self.deaths > eps)) + 1

    def betti0_curve(self, scales: List[float]) -> List[int]:
        return [self.betti0(s) for s in scales]


def h0_barcode(X: np.ndarray, k: int = 20) -> Barcode:
    """Exact degree-0 persistence barcode of the cell cloud via the MST.

    A symmetric ``k``NN distance graph is built (sufficient because every MST
    edge of a Euclidean point cloud joins ``k``NN-adjacent points for moderate
    ``k``), and its minimum spanning tree is computed; the sorted tree-edge
    weights are the finite ``H_0`` death scales.
    """
    n = X.shape[0]
    k = int(min(k, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    w = dist[:, 1:].reshape(-1)
    G = csr_matrix((w, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    mst = minimum_spanning_tree(G)
    deaths = np.sort(mst.data.astype(float))
    return Barcode(deaths=deaths)


def persistence_features(bc: Barcode) -> dict:
    """Scalar topology descriptors derived from the degree-0 barcode.

    ``max_persistence`` is the longest finite bar (the most isolated split),
    ``total_persistence`` the summed bar length, ``persistence_entropy`` the
    Shannon entropy of the normalized bar lengths (low when one split dominates),
    and ``n_significant`` the number of bars longer than ``mean + 2 sd`` -- an
    estimate of the number of well-separated sub-populations.
    """
    d = bc.deaths
    if d.size == 0:
        return {"max_persistence": 0.0, "total_persistence": 0.0,
                "persistence_entropy": 0.0, "n_significant": 0}
    p = d / (d.sum() + 1e-12)
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    thr = float(d.mean() + 2.0 * d.std())
    return {
        "max_persistence": float(d.max()),
        "total_persistence": float(d.sum()),
        "persistence_entropy": ent,
        "n_significant": int(np.sum(d > thr)),
    }


def per_cell_isolation(X: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-cell topological isolation: distance to the ``k``-th nearest neighbour.

    A cheap, persistence-flavoured per-cell signal -- large in the sparse,
    long-bar pockets where rare states live -- used by the topology-aware active
    sampler (:mod:`topocell.policy`).
    """
    n = X.shape[0]
    k = int(min(k, max(1, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nn.kneighbors(X)
    return dist[:, -1]
