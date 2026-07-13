"""Synthetic branching benchmark for the directed spectral-truncated operator.

DOMAIN NOTE (read before citing numbers). The manuscript frames this benchmark
in quantum-many-body language: a control parameter swept across a phase
transition, with "states", "device", "shot noise", and "Hamiltonian" splits.
The generator below keeps the older single-cell-lineage vocabulary
(``branch``/``state``/``donor``/``batch``/``perturbation``/``pseudotime``) because
the *method* is domain-agnostic: it classifies labelled points on a directed
similarity graph regardless of what the points physically are. The variable names
here are the ML-library convention, not a claim about biology; the runner maps
them to the manuscript's physical axes (donor->device, batch->shot-noise,
time->sweep schedule, perturbation->Hamiltonian family). These synthetic blobs are
NOT a first-principles quantum simulation -- they share only the statistical
structure (branching manifold, drift, a rare critical cluster) the classifier must
handle. Genuine validation on classical-shadow records from a programmable
simulator is stated as future work in the manuscript and is not claimed here.

Original design notes follow.

Synthetic branching lineage -- the sanity benchmark.

Real single-cell data (scRNA-seq) is high-dimensional, noisy, and confounded by
donor, batch and perturbation. We cannot ship it, and we will not fabricate it
(see ``ingest.py`` for honest hooks into real AnnData). Instead this module
generates a *fully controlled* synthetic lineage with the same structure the
method must handle, so every claim is checkable:

  * cells live in a low-dimensional feature space (a stand-in for a PCA / latent
    embedding of expression);
  * they progress along a **tree of 2-3 branches** over a continuous
    *pseudotime*, as Gaussian blobs whose mean drifts along each branch;
  * each cell carries the labels a forecasting model would actually use:
    ``branch`` / ``state`` (discrete cell state), ``pseudotime``, ``donor``,
    ``batch``, and a ``perturbation`` condition;
  * one state is deliberately **rare** (a small cluster) -- the case that
    matters biologically (a transient progenitor, a treatment-resistant clone)
    and that ordinary classifiers miss.

Everything is seeded, so the generated lineage is bit-for-bit reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Lineage:
    """A generated single-cell dataset with all metadata a split needs."""

    X: np.ndarray              # (n_cells, n_features) feature embedding
    branch: np.ndarray         # (n_cells,) int branch id along the tree
    state: np.ndarray          # (n_cells,) int discrete cell-state label
    pseudotime: np.ndarray     # (n_cells,) float in [0, 1], progression
    donor: np.ndarray          # (n_cells,) int donor id (a biological replicate)
    batch: np.ndarray          # (n_cells,) int batch id (a technical replicate)
    perturbation: np.ndarray   # (n_cells,) int condition id (0 = control)
    rare_state: int            # which `state` label is the rare one
    meta: Dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.X.shape[0]

    @property
    def n_states(self) -> int:
        return int(self.state.max()) + 1

    def state_frequencies(self) -> Dict[int, float]:
        n = self.n_cells
        return {int(s): float(np.mean(self.state == s)) for s in np.unique(self.state)}


# A small binary tree of branches over pseudotime. Each branch is (parent, t0):
# it splits off from `parent` at pseudotime `t0` and runs to t=1. Branch 0 is
# the trunk. This yields a 3-leaf lineage (a common progenitor -> two fates).
_TREE = [
    (None, 0.0),   # 0: trunk / progenitor
    (0, 0.45),     # 1: first fate, splits at t=0.45
    (0, 0.45),     # 2: second fate, splits at t=0.45
]


def _branch_mean(branch: int, t: float, n_features: int, rng: np.random.Generator,
                 anchors: np.ndarray) -> np.ndarray:
    """Mean feature vector of a cell on `branch` at pseudotime `t`.

    The trunk drifts from the origin toward a shared anchor; each fate drifts
    from the split point toward its own anchor. This makes early cells overlap
    (hard to separate) and late cells well-resolved -- exactly the regime where
    a graph smoother should beat a point classifier.
    """
    parent, t0 = _TREE[branch]
    if parent is None:                      # trunk: origin -> anchor[0]
        return anchors[0] * (t / 1.0)
    # progress along the parent up to the split, then along this branch
    split_pt = anchors[0] * (t0 / 1.0)
    frac = (t - t0) / max(1e-6, 1.0 - t0)
    return split_pt + (anchors[branch] - split_pt) * np.clip(frac, 0.0, 1.0)


def make_lineage(
    n_cells: int = 600,
    n_features: int = 8,
    n_branches: int = 3,
    n_donors: int = 4,
    n_batches: int = 2,
    n_perturbations: int = 3,
    noise: float = 0.45,
    rare_fraction: float = 0.05,
    seed: int = 0,
) -> Lineage:
    """Generate a seeded branching lineage with a rare cell state.

    Parameters mirror the knobs an experimentalist would care about: number of
    cells, donors/batches (replicate structure), perturbation conditions, and
    how rare the rare state is. ``state`` equals ``branch`` for the common
    branches; the rare state is carved out of the tip of branch 2.
    """
    rng = np.random.default_rng(seed)
    n_branches = int(np.clip(n_branches, 2, 3))

    # Fixed, well-separated anchors so the geometry is reproducible across seeds.
    anchors = np.zeros((3, n_features))
    anchors[0, 0] = 3.0                      # trunk endpoint
    anchors[1, :2] = (5.0, 4.0)              # fate 1
    anchors[2, :2] = (5.0, -4.0)             # fate 2

    # Assign cells to branches: trunk gets a share, fates split the rest.
    branch_p = np.array([0.4, 0.3, 0.3][:n_branches], dtype=float)
    branch_p = branch_p / branch_p.sum()
    branch = rng.choice(n_branches, size=n_cells, p=branch_p)

    X = np.zeros((n_cells, n_features))
    pseudotime = np.zeros(n_cells)
    for i in range(n_cells):
        b = int(branch[i])
        _, t0 = _TREE[b]
        # trunk cells span [0, 1]; fate cells live after the split point.
        lo = 0.0 if b == 0 else t0
        t = rng.uniform(lo, 1.0)
        pseudotime[i] = t
        mu = _branch_mean(b, t, n_features, rng, anchors)
        X[i] = mu + rng.normal(0.0, noise, size=n_features)

    # Discrete state starts equal to branch.
    state = branch.copy()

    # Carve a RARE state out of the late tip of branch 2 (a transient subpopulation).
    rare_state = n_branches            # a brand-new label id
    tip = (branch == (n_branches - 1)) & (pseudotime > 0.8)
    tip_idx = np.flatnonzero(tip)
    n_rare = max(3, int(round(rare_fraction * n_cells)))
    if tip_idx.size:
        chosen = rng.choice(tip_idx, size=min(n_rare, tip_idx.size), replace=False)
        state[chosen] = rare_state
        # Nudge the rare cells into their own pocket of feature space.
        X[chosen] += np.array([1.5, 1.0] + [0.0] * (n_features - 2))

    # Replicate / condition structure, assigned independently of biology so the
    # splits are non-trivial (a donor spans multiple branches, etc.).
    donor = rng.integers(0, n_donors, size=n_cells)
    batch = rng.integers(0, n_batches, size=n_cells)
    perturbation = rng.integers(0, n_perturbations, size=n_cells)
    # Perturbation shifts the feature space slightly (a treatment effect), but
    # never erases the lineage -- so a model can still forecast across conditions.
    for c in range(1, n_perturbations):
        mask = perturbation == c
        X[mask] += rng.normal(0.0, 0.15, size=n_features) + 0.3 * c

    return Lineage(
        X=X, branch=branch, state=state, pseudotime=pseudotime,
        donor=donor, batch=batch, perturbation=perturbation,
        rare_state=int(rare_state),
        meta={
            "n_branches": n_branches,
            "n_states": int(state.max()) + 1,
            "noise": noise,
            "rare_fraction": rare_fraction,
            "seed": seed,
        },
    )
