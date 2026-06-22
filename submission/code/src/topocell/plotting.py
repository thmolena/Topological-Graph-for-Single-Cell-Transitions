"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (see Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red--green trap. The
    embedding scatter keeps the perceptually uniform, colour-blind-safe
    ``viridis`` map for its continuous branch axis.
  * Error bars / 95% CI are shown wherever a mean is plotted; the caption states
    n and that the interval is a 95% confidence interval.
  * Top/right spines removed for an uncluttered Nature-style frame.

The figures are produced purely from ``results/summary.json`` -- the single
source of truth written by the experiment runner -- so they regenerate
deterministically from fixed seeds (the embedding scatter additionally
re-derives the lineage from the recorded run configuration).
"""
from __future__ import annotations

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF
# backend stamps a fixed CreationDate -> byte-identical figure PDFs across runs
# (the plotted numbers are already deterministic: they are read from the
# fixed-seed results/summary.json, the single source of truth).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# --- Okabe-Ito colour-blind-safe qualitative palette ------------------------
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Column widths in inches (Nature: single column 89 mm, double column 183 mm).
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

# Fixed display colours for the two-method comparison (Okabe-Ito).
METHOD_COLORS = {
    "graph_smoothed": NMI_PALETTE[0],  # blue
    "baseline_knn": NMI_PALETTE[7],    # black/grey -> use a neutral grey below
}
GREY = "#9E9E9E"


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so text stays selectable/editable
            "ps.fonttype": 42,
            "svg.hashsalt": "topocell",  # deterministic element IDs across runs
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=NMI_PALETTE),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


def panel_label(ax, letter: str, x: float = -0.18, y: float = 1.02) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


# --- Method-overview schematic (NMI 'Figure 1' convention) ------------------
def _box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text."""
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=7.2,
        zorder=5,
    )
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                        shrinkA=2, shrinkB=2),
    )


def fig_schematic(summary: Dict, out: Path) -> Path:
    """Method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline for the single-cell transition-forecasting method:
    a seeded synthetic branching lineage with replicate, condition and
    pseudotime metadata is embedded in feature space; a symmetric $k$NN cell
    graph is built (its connected-component / Betti-0 count monitored as a build
    sanity check); a small fraction of cells is annotated; graph-smoothed label
    propagation diffuses those labels over the graph and is compared
    apples-to-apples against a plain $k$NN classifier under four leakage-checked
    transfer splits; a topology-aware active-sampling policy targets the rare
    state. No in-plot title -- description lives in the LaTeX caption.
    """
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, green, orange, purple, grey = (
        "#D6E6F2",
        "#D6EFE3",
        "#FBE6D4",
        "#ECDCE9",
        "#ECECEC",
    )
    y = 0.40
    h = 0.34
    boxes = [
        (0.010, 0.150, "synthetic lineage\ncells $\\times$ features\nbranch, donor, batch,\npert., pseudotime", blue),
        (0.205, 0.165, "cell $k$NN graph\n$A=A^{\\top}$\n+ Betti-0 check", green),
        (0.420, 0.155, "annotate a few cells\n(label propagation\nvs plain $k$NN)", orange),
        (0.620, 0.165, "leakage-checked\ntransfer splits\n(donor/batch/time/pert.)", purple),
        (0.835, 0.155, "forecast accuracy,\nECE, rare-state\nrecall", grey),
    ]
    rights = []
    lefts = []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r)
        lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])

    # Annotate the topology-aware active-sampling branch and the rare-state goal.
    ax.text(0.4975, 0.07,
            "topology-aware active sampling: uncertainty $\\times$ inverse local "
            "density   $\\bullet$   targets the $\\sim$4% rare state",
            ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(0.4975, 0.95,
            "all data synthetic & seeded; every split passes a programmatic "
            "no-leakage check",
            ha="center", va="center", fontsize=6.8, color="#333333")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_lineage(summary: Dict, out: Path) -> Path:
    """2D branching-lineage scatter, coloured by branch, rare state highlighted.

    Regenerates the lineage from the run's config so the figure is reproducible
    from ``summary.json`` alone (no pickled arrays). The first two feature
    dimensions are the embedding plane. The continuous ``branch`` axis uses the
    perceptually uniform, colour-blind-safe viridis map; no in-plot title.
    """
    from .synthetic import make_lineage

    apply_nmi_style()
    cfg = summary["config"]
    lin = make_lineage(
        n_cells=cfg["n_cells"], n_features=cfg["n_features"], n_branches=cfg["n_branches"],
        n_donors=cfg["n_donors"], n_batches=cfg["n_batches"],
        n_perturbations=cfg["n_perturbations"], noise=cfg["noise"],
        rare_fraction=cfg["rare_fraction"], seed=cfg["seed"],
    )
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.6))
    common = lin.state != lin.rare_state
    sc = ax.scatter(lin.X[common, 0], lin.X[common, 1], c=lin.branch[common],
                    cmap="viridis", s=10, alpha=0.85, linewidths=0)
    rare = lin.state == lin.rare_state
    ax.scatter(lin.X[rare, 0], lin.X[rare, 1], c="#CC79A7", s=34, marker="*",
               edgecolors="k", linewidths=0.4,
               label=f"rare state ({rare.mean()*100:.1f}% of cells)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("branch")
    cbar.set_ticks(sorted(np.unique(lin.branch[common]).tolist()))
    cbar.outline.set_linewidth(0.8)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.legend(loc="lower right", handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_forecast_bars(summary: Dict, out: Path) -> Path:
    """Per-split accuracy and rare-state recall (graph-smoothed vs baseline)
    plus the active-vs-random rare-recall comparison, all with 95% CIs.

    Three-panel figure with bold lower-case panel labels. Error bars are the
    95% confidence interval over the n=5 seeds, taken directly from
    ``summary.json``. No in-plot titles -- descriptions live in the caption.
    """
    apply_nmi_style()
    by_split = summary["by_split"]
    splits = list(by_split.keys())
    x = np.arange(len(splits))
    w = 0.38
    blue = METHOD_COLORS["graph_smoothed"]

    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.7))

    # (a) accuracy
    ax = axes[0]
    gs = [by_split[s]["graph_smoothed"]["accuracy"]["mean"] for s in splits]
    gs_e = [by_split[s]["graph_smoothed"]["accuracy"]["ci95"] for s in splits]
    bl = [by_split[s]["baseline_knn"]["accuracy"]["mean"] for s in splits]
    bl_e = [by_split[s]["baseline_knn"]["accuracy"]["ci95"] for s in splits]
    ax.bar(x - w / 2, gs, w, yerr=gs_e, capsize=2,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           label="graph-smoothed", color=blue)
    ax.bar(x + w / 2, bl, w, yerr=bl_e, capsize=2,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           label="baseline $k$NN", color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel("forecast accuracy"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", handlelength=1.1)
    panel_label(ax, "a")

    # (b) rare-state recall
    ax = axes[1]
    gs = [by_split[s]["graph_smoothed"]["rare_recall"]["mean"] for s in splits]
    gs_e = [by_split[s]["graph_smoothed"]["rare_recall"]["ci95"] for s in splits]
    bl = [by_split[s]["baseline_knn"]["rare_recall"]["mean"] for s in splits]
    bl_e = [by_split[s]["baseline_knn"]["rare_recall"]["ci95"] for s in splits]
    ax.bar(x - w / 2, gs, w, yerr=gs_e, capsize=2,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           label="graph-smoothed", color=blue)
    ax.bar(x + w / 2, bl, w, yerr=bl_e, capsize=2,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           label="baseline $k$NN", color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel("rare-state recall"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", handlelength=1.1)
    panel_label(ax, "b")

    # (c) active vs random sampling rare recall
    ax = axes[2]
    act = summary["active_sampling"]
    vals = [act["topology"]["rare_recall"]["mean"], act["random"]["rare_recall"]["mean"]]
    errs = [act["topology"]["rare_recall"]["ci95"], act["random"]["rare_recall"]["ci95"]]
    ax.bar(["topology\n(active)", "random"], vals, yerr=errs, capsize=3,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           color=[blue, GREY])
    ax.set_ylabel("rare-state recall after query"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
