"""Train/test splits that test the claims that matter for phase classifiers.

The headline generalization claims are *transfer across hardware and physical
nuisance variables*. A classifier that only works when train and test share
devices, shot-noise batches, Hamiltonian conditions, or control values is not a
forecaster -- it is a memorizer. So we evaluate under four split protocols and,
for the grouping splits, **assert in the test suite that no group appears in both
train and test** (a strict family-held-out no-leakage check):

  device        : hold out whole devices       (hardware-replicate transfer)
  shot_batch    : hold out whole shot batches  (shot-noise replicate transfer)
  schedule      : train on EARLY control,       (extrapolate the sweep:
                  test on LATE control            classify the next phase)
  hamiltonian   : hold out a whole condition   (classify an unseen perturbation)

Each split returns boolean ``train``/``test`` masks over the state samples of a
``PhaseSweep``, plus a ``labeled`` mask: the small *annotated* subset of the
train side. ``has_leakage`` is the gate used by the runner and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .synthetic import PhaseSweep

SPLIT_TYPES = ["device", "shot_batch", "schedule", "hamiltonian"]


@dataclass
class Split:
    kind: str
    train: np.ndarray          # boolean mask over states (the candidate pool)
    test: np.ndarray           # boolean mask over states (held out, never annotated)
    labeled: np.ndarray        # boolean mask: the annotated subset of `train`
    held_out: object           # the held-out group id (or a control cutoff)
    group_of_state: np.ndarray  # the grouping variable per state (for leakage checks)


def _grouping(ps: PhaseSweep, kind: str) -> np.ndarray:
    if kind == "device":
        return ps.device
    if kind == "shot_batch":
        return ps.shot_batch
    if kind == "hamiltonian":
        return ps.hamiltonian
    if kind == "schedule":
        # not a categorical group; handled separately, but return a coarse bin
        return (ps.control >= 0.5).astype(int)
    raise ValueError(f"unknown split kind {kind!r}")


def _annotate(train: np.ndarray, ps: PhaseSweep, label_fraction: float,
              seed: int) -> np.ndarray:
    """Pick the annotated subset of the train pool (a fraction of train states).

    Lightly stratified so the rare critical regime is represented when present in
    train -- with zero rare labels *no* method could recall it (the comparison
    would be uninformative rather than honest).
    """
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train)
    n_label = max(ps.n_phases, int(round(label_fraction * train_idx.size)))
    n_label = min(n_label, train_idx.size)

    labeled = np.zeros(train.size, dtype=bool)
    chosen = []
    for s in np.unique(ps.phase[train]):
        members = train_idx[ps.phase[train_idx] == s]
        take = min(2, members.size)
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    chosen = set(chosen)
    remaining = [i for i in train_idx.tolist() if i not in chosen]
    rng.shuffle(remaining)
    for i in remaining:
        if len(chosen) >= n_label:
            break
        chosen.add(i)
    labeled[list(chosen)] = True
    return labeled


def make_split(ps: PhaseSweep, kind: str, held_out=None, schedule_cutoff: float = 0.7,
               label_fraction: float = 0.1, seed: int = 0) -> Split:
    """Build one split.

    For grouping splits (device/shot_batch/hamiltonian) ``held_out`` names the
    group placed entirely in the test set (default: the last group id). For the
    ``schedule`` split, states with control < ``schedule_cutoff`` are train and
    the rest are test -- a genuine forward extrapolation across the sweep. In
    every case only ``label_fraction`` of the train pool is actually annotated.
    """
    if kind == "schedule":
        test = ps.control >= schedule_cutoff
        train = ~test
        labeled = _annotate(train, ps, label_fraction, seed)
        return Split(kind, train, test, labeled, schedule_cutoff, _grouping(ps, kind))

    group = _grouping(ps, kind)
    if held_out is None:
        held_out = int(np.max(group))
    test = group == held_out
    train = ~test
    labeled = _annotate(train, ps, label_fraction, seed)
    return Split(kind, train, test, labeled, held_out, group)


def has_leakage(split: Split) -> bool:
    """True if any held-out group leaks across the train/test boundary.

    For grouping splits, a clean split has *disjoint* group sets in train vs
    test. For the schedule split there is no grouping variable to leak (the cut
    is on a continuous coordinate), so it is clean by construction.
    """
    if split.kind == "schedule":
        return False
    train_groups = set(np.unique(split.group_of_state[split.train]).tolist())
    test_groups = set(np.unique(split.group_of_state[split.test]).tolist())
    return len(train_groups & test_groups) > 0


def all_splits(ps: PhaseSweep, kinds: List[str] = None,
               label_fraction: float = 0.1, seed: int = 0) -> Dict[str, Split]:
    """All requested splits keyed by kind (defaults to the four protocols)."""
    kinds = kinds or SPLIT_TYPES
    return {k: make_split(ps, k, label_fraction=label_fraction, seed=seed)
            for k in kinds}
