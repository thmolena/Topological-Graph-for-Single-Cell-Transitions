"""Split integrity: donor / batch / perturbation splits must NOT leak groups.

Family-held-out no-leakage test: no held-out group may appear in
both the train and test sets of a grouping split.
"""
import numpy as np

from stdd.splits import SPLIT_TYPES, all_splits, has_leakage, make_split
from stdd.synthetic import make_lineage


def test_grouping_splits_have_no_leakage():
    lin = make_lineage(n_cells=500, n_donors=4, n_batches=2, n_perturbations=3, seed=0)
    for kind in ("donor", "batch", "perturbation"):
        split = make_split(lin, kind)
        assert not has_leakage(split), kind
        # held-out group is entirely in test, never in train.
        group = split.group_of_cell
        assert np.all(group[split.test] == split.held_out)
        assert np.all(group[split.train] != split.held_out)


def test_labeled_is_a_subset_of_train_and_excludes_held_out():
    lin = make_lineage(n_cells=500, n_donors=4, seed=3)
    for kind in ("donor", "batch", "perturbation"):
        split = make_split(lin, kind, label_fraction=0.1, seed=3)
        # annotated cells are always inside the train pool, never in test.
        assert np.all(split.train[split.labeled])
        assert not np.any(split.test[split.labeled])
        # the held-out group is never annotated (no leakage into supervision).
        assert split.held_out not in set(np.unique(split.group_of_cell[split.labeled]).tolist())
        # the budget is a strict fraction of train, not all of it.
        assert split.labeled.sum() < split.train.sum()


def test_time_split_is_forward_in_time():
    lin = make_lineage(n_cells=500, seed=1)
    split = make_split(lin, "time", time_cutoff=0.5)
    # train is strictly earlier pseudotime than test (a real extrapolation).
    assert lin.pseudotime[split.train].max() < lin.pseudotime[split.test].min() + 1e-9
    assert not has_leakage(split)


def test_all_splits_present_and_partition():
    lin = make_lineage(n_cells=400, seed=2)
    splits = all_splits(lin)
    assert set(splits.keys()) == set(SPLIT_TYPES)
    for kind, split in splits.items():
        # train/test are a disjoint cover of all cells.
        assert np.all(split.train ^ split.test)
        assert split.train.sum() > 0 and split.test.sum() > 0
