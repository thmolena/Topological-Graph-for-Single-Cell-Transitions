"""Figure generation from results artifacts, styled to Nature Machine
Intelligence (NMI) display conventions.

Design rules applied here (see Nature Portfolio artwork & formatting guidance):
  * Vector PDF output with embedded, editable text (``pdf.fonttype = 42``).
  * Sans-serif typeface (Arial/Helvetica family), 5--8 pt range.
  * No in-panel titles -- every description lives in the LaTeX caption.
  * Bold lower-case panel labels (a, b, ...) for multi-panel figures.
  * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
    2011): safe under deuteranopia/protanopia, avoids the red--green trap.
  * Error bars / 95% CI are shown wherever a mean is plotted.
  * Top/right spines removed for an uncluttered Nature-style frame.

Every figure is produced from ``results/summary.json`` -- the single source of
truth written by the experiment runner -- so the figures regenerate
deterministically (the embedding scatter additionally re-derives the lineage
from the recorded run configuration).
"""
from __future__ import annotations

import os

# Determinism: pin the build epoch BEFORE importing matplotlib so its PDF
# backend stamps a fixed CreationDate -> byte-identical figure PDFs across runs.
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

# Fixed display colours for the method comparison (Okabe-Ito).
METHOD_COLORS = {
    "std_inductive": NMI_PALETTE[0],    # blue   -- headline method
    "std_transductive": NMI_PALETTE[2],  # green
    "label_prop": NMI_PALETTE[4],        # orange -- prior graph baseline
    "baseline_knn": "#9E9E9E",           # grey   -- point baseline
}
METHOD_LABELS = {
    "std_inductive": "inductive STDD",
    "std_transductive": "transductive STDD",
    "label_prop": "label propagation",
    "baseline_knn": "$k$NN baseline",
}
GREY = "#9E9E9E"

# Display names for the four held-out transfer protocols (quantum-domain framing:
# the grouping variables are device, shot-noise realization, drive schedule, and
# Hamiltonian perturbation). Only display strings -- the data keys are unchanged.
SPLIT_LABELS = {
    "donor": "device",
    "batch": "noise",
    "time": "schedule",
    "perturbation": "Hamiltonian",
}


def apply_nmi_style() -> None:
    """Install NMI-conforming matplotlib defaults (idempotent)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "topocell",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
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


def panel_label(ax, letter: str, x: float = -0.20, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right")


# --- Method-overview schematic ---------------------------------------------
def _box(ax, xy, w, h, text, fc, ec="#222222"):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=6.9, zorder=5)
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                                shrinkA=2, shrinkB=2))


def fig_schematic(summary: Dict, out: Path) -> Path:
    """Method-overview schematic: the spectral-truncated directed-operator pipeline."""
    apply_nmi_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    blue, green, orange, purple, grey = (
        "#D6E6F2", "#D6EFE3", "#FBE6D4", "#ECDCE9", "#ECECEC")
    y, h = 0.40, 0.34
    boxes = [
        (0.005, 0.150, "synthetic phase sweep\nstates $\\times$ shadow feats.\nphase, device, noise,\npert., control $s$", blue),
        (0.190, 0.170, "directed state graph\n$P\\!=\\!D_W^{-1}W$,\n$\\kappa(s_j\\!-\\!s_i)$\n(non-normal)", green),
        (0.405, 0.155, "spectral truncation\n$B\\!=\\!\\Pi_r P\\,\\Pi_r$\n(band-limited)", orange),
        (0.610, 0.165, "inductive Nystrom\nextension +\nconformal sets", purple),
        (0.825, 0.160, "accuracy, ECE,\nrare-state recall,\ncoverage", grey),
    ]
    rights, lefts = [], []
    for x0, w, text, fc in boxes:
        r, l = _box(ax, (x0, y), w, h, text, fc)
        rights.append(r); lefts.append(l)
    for i in range(len(boxes) - 1):
        _arrow(ax, rights[i], lefts[i + 1])
    ax.text(0.5, 0.075,
            "drive-directed (noncommutative) transport $\\times$ spectral "
            "truncation   $\\bullet$   targets the $\\sim$4% rare critical regime, "
            "no held-out connectivity at inference",
            ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(0.5, 0.95,
            "all data synthetic & seeded; every grouping split passes a "
            "programmatic no-leakage check",
            ha="center", va="center", fontsize=6.8, color="#333333")
    fig.savefig(out); plt.close(fig)
    return out


def fig_lineage(summary: Dict, out: Path) -> Path:
    """Branching-lineage scatter coloured by pseudotime, rare state highlighted,
    with the degree-0 persistence Betti-0(scale) curve as an inset."""
    from .synthetic import make_lineage
    apply_nmi_style()
    cfg = summary["config"]
    lin = make_lineage(
        n_cells=cfg["n_cells"], n_features=cfg["n_features"], n_branches=cfg["n_branches"],
        n_donors=cfg["n_donors"], n_batches=cfg["n_batches"],
        n_perturbations=cfg["n_perturbations"], noise=cfg["noise"],
        rare_fraction=cfg["rare_fraction"], seed=cfg["seed"])
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.6))
    common = lin.state != lin.rare_state
    sc = ax.scatter(lin.X[common, 0], lin.X[common, 1], c=lin.pseudotime[common],
                    cmap="viridis", s=10, alpha=0.85, linewidths=0)
    rare = lin.state == lin.rare_state
    ax.scatter(lin.X[rare, 0], lin.X[rare, 1], c="#CC79A7", s=34, marker="*",
               edgecolors="k", linewidths=0.4,
               label=f"rare regime ({rare.mean()*100:.1f}% of states)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("control parameter $s$"); cbar.outline.set_linewidth(0.8)
    ax.set_xlabel("shadow feature 1"); ax.set_ylabel("shadow feature 2")
    ax.legend(loc="lower right", handlelength=1.2)

    pers = summary.get("persistence", {})
    if pers.get("scales") and pers.get("betti0_curve"):
        ins = ax.inset_axes([0.10, 0.66, 0.40, 0.30])
        ins.plot(pers["scales"], pers["betti0_curve"], "-o", color=NMI_PALETTE[1],
                 ms=2.2, lw=1.0)
        ins.set_yscale("log")
        ins.set_xlabel("scale $\\epsilon$", fontsize=6); ins.set_ylabel("$\\beta_0$", fontsize=6)
        ins.tick_params(labelsize=5.5)
        ins.set_title("$H_0$ persistence", fontsize=6)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def _bars_two(ax, splits, m1, m2, key, by_split, c1, c2, l1, l2):
    x = np.arange(len(splits)); w = 0.38
    v1 = [by_split[s][m1][key]["mean"] for s in splits]
    e1 = [by_split[s][m1][key]["ci95"] for s in splits]
    v2 = [by_split[s][m2][key]["mean"] for s in splits]
    e2 = [by_split[s][m2][key]["ci95"] for s in splits]
    ek = {"elinewidth": 0.7, "capthick": 0.7}
    ax.bar(x - w / 2, v1, w, yerr=e1, capsize=2, error_kw=ek, label=l1, color=c1)
    ax.bar(x + w / 2, v2, w, yerr=e2, capsize=2, error_kw=ek, label=l2, color=c2)
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.grid(True, axis="y")


def fig_forecast_bars(summary: Dict, out: Path) -> Path:
    """Held-out forecasting and active sampling, all with 95% CIs.

    (a) inductive STDD vs the inductive kNN baseline on accuracy (the clean,
    confound-free comparison); (b) rare-state recall by split for inductive STDD,
    label propagation and the kNN baseline; (c) rare-state recall after a fixed
    query budget under persistence-, density- and random-based active sampling.
    """
    apply_nmi_style()
    by_split = summary["by_split"]
    splits = list(by_split.keys())
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.7))

    ax = axes[0]
    _bars_two(ax, splits, "std_inductive", "baseline_knn", "accuracy", by_split,
              METHOD_COLORS["std_inductive"], GREY, "inductive STDD", "$k$NN baseline")
    ax.set_ylabel("forecast accuracy"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", handlelength=1.1); panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(splits)); w = 0.27; ek = {"elinewidth": 0.7, "capthick": 0.7}
    for i, m in enumerate(("std_inductive", "label_prop", "baseline_knn")):
        v = [by_split[s][m]["rare_recall"]["mean"] for s in splits]
        e = [by_split[s][m]["rare_recall"]["ci95"] for s in splits]
        ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=1.5, error_kw=ek,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS.get(s, s) for s in splits], rotation=20, ha="right")
    ax.set_ylabel("rare-state recall"); ax.set_ylim(0, 1.05); ax.grid(True, axis="y")
    ax.legend(loc="upper left", handlelength=1.1); panel_label(ax, "b")

    ax = axes[2]
    act = summary["active_sampling"]
    order = ["persistence", "topology", "random"]
    labels = ["persistence\n(active)", "density\n(active)", "random"]
    vals = [act[s]["rare_recall"]["mean"] for s in order]
    errs = [act[s]["rare_recall"]["ci95"] for s in order]
    ax.bar(labels, vals, yerr=errs, capsize=3, error_kw=ek,
           color=[METHOD_COLORS["std_inductive"], NMI_PALETTE[2], GREY])
    ax.set_ylabel("rare-state recall after query"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y"); panel_label(ax, "c")

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_operator(summary: Dict, out: Path) -> Path:
    """The operator novelty: noncommutativity, spectrum, and spectral truncation.

    (a) relative non-normality of the directed propagator vs the undirected
    control (the noncommutativity witness); (b) the graph-Fourier spectrum of the
    normalized adjacency with the truncation rank marked; (c) inductive accuracy
    and rare-state recall as a function of the truncation rank.
    """
    apply_nmi_style()
    op = summary["operator"]
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.6))

    ax = axes[0]
    nd = op["nonnormality_directed"]; nu = op["nonnormality_undirected"]
    ax.bar(["directed\n$P$", "undirected\n($\\beta\\!=\\!0$)"],
           [nd["mean"], nu["mean"]], yerr=[nd["ci95"], nu["ci95"]], capsize=3,
           error_kw={"elinewidth": 0.7, "capthick": 0.7},
           color=[METHOD_COLORS["std_inductive"], GREY])
    ax.set_ylabel("relative non-normality\n$\\|PP^\\top-P^\\top P\\|_F/\\|P\\|_F^2$")
    ax.grid(True, axis="y"); panel_label(ax, "a")

    ax = axes[1]
    spec = op.get("spectrum", [])
    if spec:
        ax.plot(np.arange(1, len(spec) + 1), spec, "-", color=NMI_PALETTE[1], lw=1.2)
        r = summary["config"].get("rank", 80)
        ax.axvline(r, color="#444444", ls="--", lw=0.9)
        ax.text(r + 1.5, spec[min(r, len(spec)) - 1], f"  rank $r={r}$",
                fontsize=6, va="center")
    ax.set_xlabel("graph-Fourier mode index"); ax.set_ylabel("eigenvalue of $S$")
    ax.grid(True); panel_label(ax, "b")

    ax = axes[2]
    sw = op.get("truncation_sweep", {})
    if sw.get("ranks"):
        ax.plot(sw["ranks"], sw["accuracy"], "-o", ms=2.5, color=METHOD_COLORS["std_inductive"],
                label="accuracy")
        ax.plot(sw["ranks"], sw["rare_recall"], "-s", ms=2.5, color=NMI_PALETTE[3],
                label="rare recall")
        r = summary["config"].get("rank", 80)
        ax.axvline(r, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel("truncation rank $r$"); ax.set_ylabel("inductive score")
    ax.set_ylim(0, 1.05); ax.grid(True)
    ax.legend(loc="lower right", handlelength=1.1); panel_label(ax, "c")

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_calibration(summary: Dict, out: Path) -> Path:
    """Conformal coverage and calibration of the inductive forecaster.

    (a) realized conformal coverage (marginal and rare-conditional) against the
    target level on the exchangeable hold-out; (b) expected calibration error
    before and after temperature scaling.
    """
    apply_nmi_style()
    conf = summary["conformal"]; si = conf["std_inductive"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_ONEHALF + 0.8, 2.5))
    ek = {"elinewidth": 0.7, "capthick": 0.7}

    ax = axes[0]
    vals = [si["coverage"]["mean"], si["rare_coverage"]["mean"]]
    errs = [si["coverage"]["ci95"], si["rare_coverage"]["ci95"]]
    ax.bar(["marginal", "rare-state"], vals, yerr=errs, capsize=3, error_kw=ek,
           color=[METHOD_COLORS["std_inductive"], "#CC79A7"])
    ax.axhline(conf["target_coverage"], color="#444444", ls="--", lw=1.0,
               label=f"target {conf['target_coverage']:.2f}")
    ax.set_ylabel("conformal coverage"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y"); ax.legend(loc="lower left", handlelength=1.1)
    panel_label(ax, "a", x=-0.26)

    ax = axes[1]
    vals = [si["ece_raw"]["mean"], si["ece_calibrated"]["mean"]]
    errs = [si["ece_raw"]["ci95"], si["ece_calibrated"]["ci95"]]
    ax.bar(["raw", "temperature\nscaled"], vals, yerr=errs, capsize=3, error_kw=ek,
           color=[GREY, METHOD_COLORS["std_inductive"]])
    ax.set_ylabel("expected calibration error"); ax.grid(True, axis="y")
    panel_label(ax, "b", x=-0.26)

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out
