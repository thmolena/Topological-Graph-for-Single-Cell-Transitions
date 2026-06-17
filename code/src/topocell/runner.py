"""End-to-end experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. The protocol, for each random seed and each split protocol
(donor / batch / time / perturbation):

  1. generate the synthetic branching lineage (seeded);
  2. build the kNN cell graph and record its Betti-0 (component) curve;
  3. forecast held-out cell states two ways -- graph-smoothed label propagation
     (the method) vs a plain kNN classifier (the baseline) -- and score
     accuracy, ECE and rare-state recall;
  4. run active sampling under a labelling budget -- topology-aware (the method)
     vs random (the baseline) -- and score the rare-state-recall lift.

Aggregates across seeds. Headline: graph-smoothed accuracy, the accuracy gain
over the baseline, and the rare-state-recall gain from active sampling. Integrity
flags assert the splits are leakage-free and Betti-0 is monotone in ``k``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import metrics
from .config import Config
from .graph import betti0_curve
from .policy import (
    KNNForecaster,
    LabelPropagationForecaster,
    active_sample,
    recall_after_query,
)
from .seed import RunProvenance, set_seed
from .splits import all_splits, has_leakage
from .synthetic import make_lineage

FORECASTERS = ["graph_smoothed", "baseline_knn"]


def _eval_forecasters(lin, split, k) -> Dict[str, Dict[str, float]]:
    """Accuracy / ECE / rare-state recall for both forecasters on one split.

    Both models see the *same* small annotated set (``split.labeled``) and
    predict the held-out ``split.test`` cells -- a semi-supervised forecast.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in FORECASTERS:
        model = (LabelPropagationForecaster(k=k) if name == "graph_smoothed"
                 else KNNForecaster(k=k))
        fc = model.predict(lin.X, lin.state, split.labeled, split.test)
        y_test = lin.state[split.test]
        out[name] = {
            "accuracy": metrics.accuracy(y_test, fc.pred),
            "ece": metrics.expected_calibration_error(y_test, fc.pred, fc.confidence),
            "rare_recall": metrics.rare_state_recall(y_test, fc.pred, lin.rare_state),
        }
    return out


def _active_sampling_lift(lin, k, budget, rng) -> Dict[str, float]:
    """Rare-state-recall lift from topology-aware vs random active sampling.

    Start from a small random seed of labels, query ``budget`` more cells under
    each strategy, and measure the recall on the still-unlabeled cells.
    """
    n = lin.n_cells
    seed_mask = np.zeros(n, dtype=bool)
    seed_idx = rng.choice(n, size=max(10, n // 10), replace=False)
    seed_mask[seed_idx] = True

    results = {}
    for strategy in ("topology", "random"):
        queried = active_sample(lin.X, lin.state, seed_mask, budget, k=k,
                                strategy=strategy, rng=np.random.default_rng(rng.integers(1 << 30)))
        _, after = recall_after_query(lin.X, lin.state, seed_mask, queried,
                                      lin.rare_state, k=k)
        # fraction of the queried cells that were actually rare (query precision)
        rare_hit = float(np.mean(lin.state[queried] == lin.rare_state)) if queried.size else 0.0
        results[strategy] = {"rare_recall_after": after, "rare_query_rate": rare_hit}
    return results


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    set_seed(cfg.seed)

    # Per-split, per-forecaster metric accumulators across seeds.
    acc: Dict[str, Dict[str, List[float]]] = {
        s: {f: [] for f in FORECASTERS} for s in cfg.splits
    }
    ece: Dict[str, Dict[str, List[float]]] = {
        s: {f: [] for f in FORECASTERS} for s in cfg.splits
    }
    rare: Dict[str, Dict[str, List[float]]] = {
        s: {f: [] for f in FORECASTERS} for s in cfg.splits
    }
    active_topo, active_rand = [], []
    active_topo_qrate, active_rand_qrate = [], []
    splits_clean = True
    betti_monotonic = True
    rare_fracs: List[float] = []
    n_states_seen: List[int] = []
    betti_curves: List[List[int]] = []

    ks_for_betti = [2, 4, 8, 16, 32]

    for s in range(cfg.n_seeds):
        seed = cfg.seed + s
        lin = make_lineage(
            n_cells=cfg.n_cells, n_features=cfg.n_features, n_branches=cfg.n_branches,
            n_donors=cfg.n_donors, n_batches=cfg.n_batches,
            n_perturbations=cfg.n_perturbations, noise=cfg.noise,
            rare_fraction=cfg.rare_fraction, seed=seed,
        )
        rare_fracs.append(lin.state_frequencies().get(lin.rare_state, 0.0))
        n_states_seen.append(lin.n_states)

        # Betti-0 monotonicity over k (a real topological property).
        ks = [k for k in ks_for_betti if k < lin.n_cells]
        curve = betti0_curve(lin.X, ks)
        betti_curves.append(curve)
        if any(curve[i + 1] > curve[i] for i in range(len(curve) - 1)):
            betti_monotonic = False

        splits = all_splits(lin, cfg.splits, label_fraction=cfg.label_fraction, seed=seed)
        for kind, split in splits.items():
            if has_leakage(split):
                splits_clean = False
            res = _eval_forecasters(lin, split, cfg.k)
            for f in FORECASTERS:
                acc[kind][f].append(res[f]["accuracy"])
                ece[kind][f].append(res[f]["ece"])
                rare[kind][f].append(res[f]["rare_recall"])

        rng = np.random.default_rng(seed + 12345)
        lift = _active_sampling_lift(lin, cfg.k, cfg.active_budget, rng)
        active_topo.append(lift["topology"]["rare_recall_after"])
        active_rand.append(lift["random"]["rare_recall_after"])
        active_topo_qrate.append(lift["topology"]["rare_query_rate"])
        active_rand_qrate.append(lift["random"]["rare_query_rate"])

    # ---- aggregate -------------------------------------------------------- #
    by_split: Dict[str, Dict] = {}
    for kind in cfg.splits:
        by_split[kind] = {}
        for f in FORECASTERS:
            by_split[kind][f] = {
                "accuracy": metrics.summarize(acc[kind][f]),
                "ece": metrics.summarize(ece[kind][f]),
                "rare_recall": metrics.summarize(rare[kind][f]),
            }

    active = {
        "topology": {
            "rare_recall": metrics.summarize(active_topo),
            "rare_query_rate": metrics.summarize(active_topo_qrate),
        },
        "random": {
            "rare_recall": metrics.summarize(active_rand),
            "rare_query_rate": metrics.summarize(active_rand_qrate),
        },
    }

    # Mean accuracy / rare-recall across splits, per forecaster (for the headline).
    def _mean_over_splits(table, metric, forecaster):
        vals = [table[k][forecaster][metric]["mean"] for k in cfg.splits]
        return float(np.mean(vals))

    gs_acc = _mean_over_splits(by_split, "accuracy", "graph_smoothed")
    bl_acc = _mean_over_splits(by_split, "accuracy", "baseline_knn")
    gs_rare = _mean_over_splits(by_split, "rare_recall", "graph_smoothed")
    bl_rare = _mean_over_splits(by_split, "rare_recall", "baseline_knn")

    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "integrity": {
            "donor_split_clean": bool(splits_clean),
            "betti_monotonic": bool(betti_monotonic),
            "rare_state_present": bool(np.mean(rare_fracs) < 0.15 and np.mean(rare_fracs) > 0),
        },
        "by_split": by_split,
        "active_sampling": active,
        "betti": {"ks": [k for k in ks_for_betti if k < cfg.n_cells],
                  "components": betti_curves[0] if betti_curves else []},
        "headline": {},
    }

    summary["headline"] = {
        "graph_smoothed_accuracy": round(gs_acc, 6),
        "baseline_accuracy": round(bl_acc, 6),
        "accuracy_gain_vs_baseline": round(gs_acc - bl_acc, 6),
        "graph_smoothed_rare_recall": round(gs_rare, 6),
        "baseline_rare_recall": round(bl_rare, 6),
        "active_rare_recall": active["topology"]["rare_recall"]["mean"],
        "random_rare_recall": active["random"]["rare_recall"]["mean"],
        "active_rare_recall_gain": round(
            active["topology"]["rare_recall"]["mean"] - active["random"]["rare_recall"]["mean"], 6
        ),
        "n_cells": int(cfg.n_cells),
        "n_states": int(np.max(n_states_seen)) if n_states_seen else 0,
        "rare_state_fraction": round(float(np.mean(rare_fracs)), 6),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
