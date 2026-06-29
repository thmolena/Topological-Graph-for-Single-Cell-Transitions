"""Synthetic phase sweep -- the sanity benchmark.

Real classical-shadow data from a programmable quantum simulator is
high-dimensional, noisy, and confounded by device, shot noise and Hamiltonian
miscalibration. We cannot ship it, and we will not fabricate it (see
``ingest.py`` for honest hooks into real shadow datasets). Instead this module
generates a *fully controlled* synthetic phase sweep with the same structure the
method must handle, so every claim is checkable:

  * quantum-state samples live in a low-dimensional feature space (a stand-in for
    a set of classical-shadow observables / a latent embedding of the snapshots);
  * they progress along a **tree of 2-3 phase branches** over a continuous
    *control parameter* (the swept drive coordinate), as Gaussian blobs whose
    mean drifts along each branch and bifurcates at a critical control value;
  * each sample carries the labels a phase classifier would actually use:
    ``phase_branch`` / ``phase`` (discrete phase label), ``control``, ``device``,
    ``shot_batch``, and a ``hamiltonian`` (perturbation) condition;
  * one phase is deliberately **rare** (a small cluster) -- the *critical regime*
    near the transition that matters physically and that ordinary classifiers
    miss.

Everything is seeded, so the generated sweep is bit-for-bit reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PhaseSweep:
    """A generated phase-sweep dataset with all metadata a split needs."""

    X: np.ndarray              # (n_states, n_features) shadow-feature embedding
    phase_branch: np.ndarray   # (n_states,) int branch id along the phase tree
    phase: np.ndarray          # (n_states,) int discrete phase label
    control: np.ndarray        # (n_states,) float in [0, 1], control parameter
    device: np.ndarray         # (n_states,) int device id (a hardware replicate)
    shot_batch: np.ndarray     # (n_states,) int shot-noise batch id
    hamiltonian: np.ndarray    # (n_states,) int Hamiltonian-perturbation id (0 = reference)
    rare_regime: int           # which `phase` label is the rare critical regime
    meta: Dict = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.X.shape[0]

    @property
    def n_phases(self) -> int:
        return int(self.phase.max()) + 1

    def phase_frequencies(self) -> Dict[int, float]:
        n = self.n_states
        return {int(s): float(np.mean(self.phase == s)) for s in np.unique(self.phase)}


# A small binary tree of phase branches over the control parameter. Each branch
# is (parent, t0): it bifurcates off `parent` at control `t0` and runs to t=1.
# Branch 0 is the trunk. This yields a 3-leaf sweep (one phase -> two fates).
_TREE = [
    (None, 0.0),   # 0: trunk / disordered phase
    (0, 0.45),     # 1: first ordered phase, bifurcates at t=0.45
    (0, 0.45),     # 2: second ordered phase, bifurcates at t=0.45
]


def _phase_branch_mean(branch: int, t: float, n_features: int, rng: np.random.Generator,
                       anchors: np.ndarray) -> np.ndarray:
    """Mean feature vector of a state on `branch` at control `t`.

    The trunk drifts from the origin toward a shared anchor; each ordered phase
    drifts from the bifurcation point toward its own anchor. Early states overlap
    (hard to separate near the critical point) and late states are well resolved
    -- exactly the regime where a graph smoother should beat a point classifier.
    """
    parent, t0 = _TREE[branch]
    if parent is None:                      # trunk: origin -> anchor[0]
        return anchors[0] * (t / 1.0)
    split_pt = anchors[0] * (t0 / 1.0)
    frac = (t - t0) / max(1e-6, 1.0 - t0)
    return split_pt + (anchors[branch] - split_pt) * np.clip(frac, 0.0, 1.0)


def make_phase_sweep(
    n_states: int = 600,
    n_features: int = 8,
    n_phase_branches: int = 3,
    n_devices: int = 4,
    n_shot_batches: int = 2,
    n_hamiltonians: int = 3,
    noise: float = 0.45,
    rare_fraction: float = 0.05,
    seed: int = 0,
) -> PhaseSweep:
    """Generate a seeded phase sweep with a rare critical regime.

    Parameters mirror the knobs an experimentalist would care about: number of
    state samples, devices/shot-noise batches (replicate structure), Hamiltonian
    conditions, and how rare the critical regime is. ``phase`` equals
    ``phase_branch`` for the common phases; the rare regime is carved out of the
    late tip of branch 2 (near the transition's far side).
    """
    rng = np.random.default_rng(seed)
    n_phase_branches = int(np.clip(n_phase_branches, 2, 3))

    # Fixed, well-separated anchors so the geometry is reproducible across seeds.
    anchors = np.zeros((3, n_features))
    anchors[0, 0] = 3.0                      # trunk endpoint
    anchors[1, :2] = (5.0, 4.0)              # ordered phase 1
    anchors[2, :2] = (5.0, -4.0)             # ordered phase 2

    branch_p = np.array([0.4, 0.3, 0.3][:n_phase_branches], dtype=float)
    branch_p = branch_p / branch_p.sum()
    phase_branch = rng.choice(n_phase_branches, size=n_states, p=branch_p)

    X = np.zeros((n_states, n_features))
    control = np.zeros(n_states)
    for i in range(n_states):
        b = int(phase_branch[i])
        _, t0 = _TREE[b]
        lo = 0.0 if b == 0 else t0
        t = rng.uniform(lo, 1.0)
        control[i] = t
        mu = _phase_branch_mean(b, t, n_features, rng, anchors)
        X[i] = mu + rng.normal(0.0, noise, size=n_features)

    # Discrete phase starts equal to branch.
    phase = phase_branch.copy()

    # Carve a RARE critical regime out of the late tip of branch 2.
    rare_regime = n_phase_branches            # a brand-new label id
    tip = (phase_branch == (n_phase_branches - 1)) & (control > 0.8)
    tip_idx = np.flatnonzero(tip)
    n_rare = max(3, int(round(rare_fraction * n_states)))
    if tip_idx.size:
        chosen = rng.choice(tip_idx, size=min(n_rare, tip_idx.size), replace=False)
        phase[chosen] = rare_regime
        # Nudge the rare-regime states into their own pocket of feature space.
        X[chosen] += np.array([1.5, 1.0] + [0.0] * (n_features - 2))

    # Replicate / condition structure, assigned independently of the phase so the
    # splits are non-trivial (a device spans multiple phases, etc.).
    device = rng.integers(0, n_devices, size=n_states)
    shot_batch = rng.integers(0, n_shot_batches, size=n_states)
    hamiltonian = rng.integers(0, n_hamiltonians, size=n_states)
    # A Hamiltonian perturbation shifts feature space slightly (a detuning), but
    # never erases the phase structure -- so a model can still classify across it.
    for c in range(1, n_hamiltonians):
        mask = hamiltonian == c
        X[mask] += rng.normal(0.0, 0.15, size=n_features) + 0.3 * c

    return PhaseSweep(
        X=X, phase_branch=phase_branch, phase=phase, control=control,
        device=device, shot_batch=shot_batch, hamiltonian=hamiltonian,
        rare_regime=int(rare_regime),
        meta={
            "n_phase_branches": n_phase_branches,
            "n_phases": int(phase.max()) + 1,
            "noise": noise,
            "rare_fraction": rare_fraction,
            "seed": seed,
        },
    )
