"""The kNN graph's Betti-0 (number of connected components) must be monotone
non-increasing as k grows -- a real topological property, seeded.

Adding edges (larger k) can only merge components, never split them, so the
component count can only stay the same or drop.
"""
import numpy as np

from topocell.graph import betti0_curve, build_knn_graph, n_connected_components
from topocell.synthetic import make_lineage


def test_betti0_monotone_nonincreasing():
    lin = make_lineage(n_cells=400, n_features=8, seed=0)
    ks = [2, 4, 8, 16, 32, 64]
    curve = betti0_curve(lin.X, ks)
    for i in range(len(curve) - 1):
        assert curve[i + 1] <= curve[i], (ks[i], ks[i + 1], curve)


def test_betti0_collapses_to_one_for_large_k():
    lin = make_lineage(n_cells=300, seed=3)
    cg = build_knn_graph(lin.X, k=200)
    assert n_connected_components(cg) == 1


def test_betti0_holds_across_seeds():
    for seed in range(4):
        lin = make_lineage(n_cells=250, seed=seed)
        curve = betti0_curve(lin.X, [3, 6, 12, 24])
        assert all(curve[i + 1] <= curve[i] for i in range(len(curve) - 1)), (seed, curve)
