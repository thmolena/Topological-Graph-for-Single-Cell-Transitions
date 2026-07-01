"""Train/test splits that test the claims that matter for single-cell models.

The headline generalization claims are *transfer across biological and technical
nuisance variables*. A model that only works when train and test share donors,
batches, conditions, or timepoints is not a forecaster -- it is a memorizer. So
we evaluate under four split protocols and, for the grouping splits, **assert in
the test suite that no group appears in both train and test** (a strict
family-held-out no-leakage check):

  donor         : hold out whole donors        (biological-replicate transfer)
  batch         : hold out whole batches       (technical-replicate transfer)
  time          : train on EARLY pseudotime,   (extrapolate the trajectory:
                  test on LATE pseudotime        forecast the next state)
  perturbation  : hold out a whole condition   (forecast an unseen perturbation)

Each split returns boolean ``train``/``test`` masks over the cells of a
``Lineage``, plus a ``labeled`` mask: the small *annotated* subset of the train
side. Real single-cell studies annotate only a handful of cells and propagate
the rest, so the forecaster is trained on ``labeled`` (a fraction of ``train``)
and evaluated on ``test``. This semi-supervised, few-label regime is exactly
where graph diffusion earns its keep over a plain point classifier.
``has_leakage`` is the gate used by the runner and the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .synthetic import Lineage

SPLIT_TYPES = ["donor", "batch", "time", "perturbation"]


@dataclass
class Split:
    kind: str
    train: np.ndarray          # boolean mask over cells (the candidate pool)
    test: np.ndarray           # boolean mask over cells (held out, never annotated)
    labeled: np.ndarray        # boolean mask: the annotated subset of `train`
    held_out: object           # the held-out group id (or a pseudotime cutoff)
    group_of_cell: np.ndarray  # the grouping variable per cell (for leakage checks)


def _grouping(lin: Lineage, kind: str) -> np.ndarray:
    if kind == "donor":
        return lin.donor
    if kind == "batch":
        return lin.batch
    if kind == "perturbation":
        return lin.perturbation
    if kind == "time":
        # not a categorical group; handled separately, but return a coarse bin
        return (lin.pseudotime >= 0.5).astype(int)
    raise ValueError(f"unknown split kind {kind!r}")


def _annotate(train: np.ndarray, lin: Lineage, label_fraction: float,
              seed: int) -> np.ndarray:
    """Pick the annotated subset of the train pool (a fraction of train cells).

    Lightly stratified so the rare state is represented when present in train --
    an analyst annotating a study would not skip the rare population entirely,
    and with zero rare labels *no* method could recall it (the comparison would
    be uninformative rather than honest).
    """
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(train)
    n_label = max(lin.n_states, int(round(label_fraction * train_idx.size)))
    n_label = min(n_label, train_idx.size)

    labeled = np.zeros(train.size, dtype=bool)
    # Guarantee a couple of labels per present state (incl. the rare one).
    chosen = []
    for s in np.unique(lin.state[train]):
        members = train_idx[lin.state[train_idx] == s]
        take = min(2, members.size)
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())
    chosen = set(chosen)
    # Fill the rest uniformly at random from the train pool.
    remaining = [i for i in train_idx.tolist() if i not in chosen]
    rng.shuffle(remaining)
    for i in remaining:
        if len(chosen) >= n_label:
            break
        chosen.add(i)
    labeled[list(chosen)] = True
    return labeled


def make_split(lin: Lineage, kind: str, held_out=None, time_cutoff: float = 0.7,
               label_fraction: float = 0.1, seed: int = 0) -> Split:
    """Build one split.

    For grouping splits (donor/batch/perturbation) ``held_out`` names the group
    placed entirely in the test set (default: the last group id). For the
    ``time`` split, cells with pseudotime < ``time_cutoff`` are train and the
    rest are test -- a genuine forward-in-time extrapolation. In every case only
    ``label_fraction`` of the train pool is actually annotated (``labeled``); the
    held-out group/time is never annotated, so there is no leakage.
    """
    if kind == "time":
        test = lin.pseudotime >= time_cutoff
        train = ~test
        labeled = _annotate(train, lin, label_fraction, seed)
        return Split(kind, train, test, labeled, time_cutoff, _grouping(lin, kind))

    group = _grouping(lin, kind)
    if held_out is None:
        held_out = int(np.max(group))
    test = group == held_out
    train = ~test
    labeled = _annotate(train, lin, label_fraction, seed)
    return Split(kind, train, test, labeled, held_out, group)


def has_leakage(split: Split) -> bool:
    """True if any held-out group leaks across the train/test boundary.

    For grouping splits, a clean split has *disjoint* group sets in train vs
    test. For the time split there is no grouping variable to leak (the cut is
    on a continuous coordinate), so it is clean by construction.
    """
    if split.kind == "time":
        return False
    train_groups = set(np.unique(split.group_of_cell[split.train]).tolist())
    test_groups = set(np.unique(split.group_of_cell[split.test]).tolist())
    return len(train_groups & test_groups) > 0


def all_splits(lin: Lineage, kinds: List[str] = None,
               label_fraction: float = 0.1, seed: int = 0) -> Dict[str, Split]:
    """All requested splits keyed by kind (defaults to the four protocols)."""
    kinds = kinds or SPLIT_TYPES
    return {k: make_split(lin, k, label_fraction=label_fraction, seed=seed)
            for k in kinds}
