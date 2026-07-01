"""Figure generation from results artifacts, styled to Physical Review display
conventions (compact, vector PDF, colour-blind-safe).

Every figure is produced from ``results/summary.json`` -- the single source of
truth written by the experiment runner -- so the figures regenerate
deterministically (the phase-sweep scatter additionally re-derives the sweep from
the recorded run configuration).
"""
from __future__ import annotations

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

# Okabe-Ito colour-blind-safe qualitative palette.
PALETTE = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
COL_SINGLE = 3.50
COL_ONEHALF = 4.75
COL_DOUBLE = 7.20

METHOD_COLORS = {
    "std_inductive": PALETTE[0],
    "std_transductive": PALETTE[2],
    "label_prop": PALETTE[4],
    "baseline_knn": "#9E9E9E",
}
METHOD_LABELS = {
    "std_inductive": "inductive STDD",
    "std_transductive": "transductive STDD",
    "label_prop": "label propagation",
    "baseline_knn": "$k$NN baseline",
}
SPLIT_LABELS = {
    "device": "device",
    "shot_batch": "shot noise",
    "schedule": "schedule",
    "hamiltonian": "Hamiltonian",
}
GREY = "#9E9E9E"


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "stdd", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.3, "lines.markersize": 3.0,
        "legend.frameon": False, "axes.prop_cycle": cycler(color=PALETTE),
        "xtick.direction": "out", "ytick.direction": "out",
        "grid.linewidth": 0.5, "grid.alpha": 0.3,
    })


def panel_label(ax, letter: str, x: float = -0.20, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="right")


def _split_labels(splits):
    return [SPLIT_LABELS.get(s, s) for s in splits]


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
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    blue, green, orange, purple, grey = (
        "#D6E6F2", "#D6EFE3", "#FBE6D4", "#ECDCE9", "#ECECEC")
    y, h = 0.40, 0.34
    boxes = [
        (0.005, 0.150, "phase sweep\nshadow features\nphase, device, shot batch,\nHam., control", blue),
        (0.190, 0.170, "directed state graph\n$P\\!=\\!D_W^{-1}W$,\n$\\kappa(t_j\\!-\\!t_i)$\n(non-normal)", green),
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
            "control-directed (noncommutative) transport $\\times$ spectral truncation   "
            "$\\bullet$   targets the $\\sim$4% rare critical regime, "
            "no held-out connectivity at inference",
            ha="center", va="center", fontsize=6.6, color="#555555")
    ax.text(0.5, 0.95,
            "all data synthetic & seeded; every grouping split passes a "
            "programmatic no-leakage check",
            ha="center", va="center", fontsize=6.8, color="#333333")
    fig.savefig(out); plt.close(fig)
    return out


def fig_sweep(summary: Dict, out: Path) -> Path:
    """Phase-sweep scatter coloured by control parameter, rare regime highlighted,
    with the degree-0 persistence Betti-0(scale) curve as an inset."""
    from .synthetic import make_phase_sweep
    apply_style()
    cfg = summary["config"]
    ps = make_phase_sweep(
        n_states=cfg["n_states"], n_features=cfg["n_features"],
        n_phase_branches=cfg["n_phase_branches"], n_devices=cfg["n_devices"],
        n_shot_batches=cfg["n_shot_batches"], n_hamiltonians=cfg["n_hamiltonians"],
        noise=cfg["noise"], rare_fraction=cfg["rare_fraction"], seed=cfg["seed"])
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 3.6))
    common = ps.phase != ps.rare_regime
    sc = ax.scatter(ps.X[common, 0], ps.X[common, 1], c=ps.control[common],
                    cmap="viridis", s=10, alpha=0.85, linewidths=0)
    rare = ps.phase == ps.rare_regime
    ax.scatter(ps.X[rare, 0], ps.X[rare, 1], c="#CC79A7", s=34, marker="*",
               edgecolors="k", linewidths=0.4,
               label=f"rare regime ({rare.mean()*100:.1f}% of states)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("control parameter"); cbar.outline.set_linewidth(0.8)
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.legend(loc="lower right", handlelength=1.2)

    pers = summary.get("persistence", {})
    if pers.get("scales") and pers.get("betti0_curve"):
        ins = ax.inset_axes([0.10, 0.66, 0.40, 0.30])
        ins.plot(pers["scales"], pers["betti0_curve"], "-o", color=PALETTE[1], ms=2.2, lw=1.0)
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
    ax.set_xticks(x); ax.set_xticklabels(_split_labels(splits), rotation=20, ha="right")
    ax.grid(True, axis="y")


def fig_forecast_bars(summary: Dict, out: Path) -> Path:
    """Held-out classification and active sampling, all with 95% CIs."""
    apply_style()
    by_split = summary["by_split"]
    splits = list(by_split.keys())
    fig, axes = plt.subplots(1, 3, figsize=(COL_DOUBLE, 2.7))

    ax = axes[0]
    _bars_two(ax, splits, "std_inductive", "baseline_knn", "accuracy", by_split,
              METHOD_COLORS["std_inductive"], GREY, "inductive STDD", "$k$NN baseline")
    ax.set_ylabel("classification accuracy"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", handlelength=1.1); panel_label(ax, "a")

    ax = axes[1]
    x = np.arange(len(splits)); w = 0.27; ek = {"elinewidth": 0.7, "capthick": 0.7}
    for i, m in enumerate(("std_inductive", "label_prop", "baseline_knn")):
        v = [by_split[s][m]["rare_recall"]["mean"] for s in splits]
        e = [by_split[s][m]["rare_recall"]["ci95"] for s in splits]
        ax.bar(x + (i - 1) * w, v, w, yerr=e, capsize=1.5, error_kw=ek,
               label=METHOD_LABELS[m], color=METHOD_COLORS[m])
    ax.set_xticks(x); ax.set_xticklabels(_split_labels(splits), rotation=20, ha="right")
    ax.set_ylabel("rare-state recall"); ax.set_ylim(0, 1.05); ax.grid(True, axis="y")
    ax.legend(loc="upper left", handlelength=1.1); panel_label(ax, "b")

    ax = axes[2]
    act = summary["active_sampling"]
    order = ["persistence", "topology", "random"]
    labels = ["persistence\n(active)", "density\n(active)", "random"]
    vals = [act[s]["rare_recall"]["mean"] for s in order]
    errs = [act[s]["rare_recall"]["ci95"] for s in order]
    ax.bar(labels, vals, yerr=errs, capsize=3, error_kw=ek,
           color=[METHOD_COLORS["std_inductive"], PALETTE[2], GREY])
    ax.set_ylabel("rare-state recall after query"); ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y"); panel_label(ax, "c")

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_operator(summary: Dict, out: Path) -> Path:
    """The operator novelty: noncommutativity, spectrum, and spectral truncation."""
    apply_style()
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
        ax.plot(np.arange(1, len(spec) + 1), spec, "-", color=PALETTE[1], lw=1.2)
        r = summary["config"].get("rank", 80)
        ax.axvline(r, color="#444444", ls="--", lw=0.9)
        ax.text(r + 1.5, spec[min(r, len(spec)) - 1], f"  rank $r={r}$", fontsize=6, va="center")
    ax.set_xlabel("graph-Fourier mode index"); ax.set_ylabel("eigenvalue of $S$")
    ax.grid(True); panel_label(ax, "b")

    ax = axes[2]
    sw = op.get("truncation_sweep", {})
    if sw.get("ranks"):
        ax.plot(sw["ranks"], sw["accuracy"], "-o", ms=2.5, color=METHOD_COLORS["std_inductive"],
                label="accuracy")
        ax.plot(sw["ranks"], sw["rare_recall"], "-s", ms=2.5, color=PALETTE[3], label="rare recall")
        r = summary["config"].get("rank", 80)
        ax.axvline(r, color="#444444", ls="--", lw=0.9)
    ax.set_xlabel("truncation rank $r$"); ax.set_ylabel("inductive score")
    ax.set_ylim(0, 1.05); ax.grid(True)
    ax.legend(loc="lower right", handlelength=1.1); panel_label(ax, "c")

    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def fig_calibration(summary: Dict, out: Path) -> Path:
    """Conformal coverage and calibration of the inductive classifier."""
    apply_style()
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
