"""Figure generation from results artifacts (matplotlib, Agg backend)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def fig_lineage(summary: Dict, out: Path) -> Path:
    """2D branching-lineage scatter, colored by branch, rare state highlighted.

    Regenerates the lineage from the run's config so the figure is reproducible
    from ``summary.json`` alone (no pickled arrays). The first two feature
    dimensions are the embedding plane.
    """
    from .synthetic import make_lineage

    cfg = summary["config"]
    lin = make_lineage(
        n_cells=cfg["n_cells"], n_features=cfg["n_features"], n_branches=cfg["n_branches"],
        n_donors=cfg["n_donors"], n_batches=cfg["n_batches"],
        n_perturbations=cfg["n_perturbations"], noise=cfg["noise"],
        rare_fraction=cfg["rare_fraction"], seed=cfg["seed"],
    )
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    common = lin.state != lin.rare_state
    sc = ax.scatter(lin.X[common, 0], lin.X[common, 1], c=lin.branch[common],
                    cmap="viridis", s=14, alpha=0.85, linewidths=0)
    rare = lin.state == lin.rare_state
    ax.scatter(lin.X[rare, 0], lin.X[rare, 1], c="#d6336c", s=40, marker="*",
               edgecolors="k", linewidths=0.4,
               label=f"rare state ({rare.mean()*100:.1f}% of cells)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("branch")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("Synthetic branching lineage")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_forecast_bars(summary: Dict, out: Path) -> Path:
    """Per-split accuracy and rare-state recall: graph-smoothed vs baseline,
    plus the active-vs-random rare-recall comparison."""
    by_split = summary["by_split"]
    splits = list(by_split.keys())
    x = np.arange(len(splits))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    # (a) accuracy
    ax = axes[0]
    gs = [by_split[s]["graph_smoothed"]["accuracy"]["mean"] for s in splits]
    bl = [by_split[s]["baseline_knn"]["accuracy"]["mean"] for s in splits]
    ax.bar(x - w / 2, gs, w, label="graph-smoothed", color="#1f6feb")
    ax.bar(x + w / 2, bl, w, label="baseline kNN", color="#b8c2cc")
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("forecast accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) Accuracy by split"); ax.legend(fontsize=8)

    # (b) rare-state recall
    ax = axes[1]
    gs = [by_split[s]["graph_smoothed"]["rare_recall"]["mean"] for s in splits]
    bl = [by_split[s]["baseline_knn"]["rare_recall"]["mean"] for s in splits]
    ax.bar(x - w / 2, gs, w, label="graph-smoothed", color="#76b900")
    ax.bar(x + w / 2, bl, w, label="baseline kNN", color="#b8c2cc")
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("rare-state recall"); ax.set_ylim(0, 1.05)
    ax.set_title("(b) Rare-state recall by split"); ax.legend(fontsize=8)

    # (c) active vs random sampling rare recall
    ax = axes[2]
    act = summary["active_sampling"]
    vals = [act["topology"]["rare_recall"]["mean"], act["random"]["rare_recall"]["mean"]]
    errs = [act["topology"]["rare_recall"]["ci95"], act["random"]["rare_recall"]["ci95"]]
    ax.bar(["topology\n(active)", "random"], vals, yerr=errs, capsize=4,
           color=["#76b900", "#b8c2cc"])
    ax.set_ylabel("rare-state recall after query"); ax.set_ylim(0, 1.05)
    ax.set_title("(c) Active sampling lift")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
