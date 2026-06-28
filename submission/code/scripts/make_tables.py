#!/usr/bin/env python
"""Generate LaTeX tables from results/summary.json.

Emits ``results/main_results.tex`` (per-split accuracy / ECE / rare-recall for
the four forecasters) and ``results/extended_results.tex`` (the operator,
conformal, active-sampling and persistence extended data). Every number is read
from the single source of truth; nothing is typed by hand.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

LABELS = {
    "std_inductive": "inductive STDD",
    "std_transductive": "transductive STDD",
    "label_prop": "label propagation",
    "baseline_knn": "$k$NN baseline",
}
ORDER = ["std_inductive", "std_transductive", "label_prop", "baseline_knn"]


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("results"); out.mkdir(exist_ok=True)
    by_split = summary["by_split"]

    # ---- main per-split table -------------------------------------------- #
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"split & method & accuracy & ECE & rare-state recall \\",
             r"\midrule"]
    for split, m in by_split.items():
        for method in ORDER:
            d = m[method]
            lines.append(f"{split} & {LABELS[method]} & {d['accuracy']['mean']:.4f} & "
                         f"{d['ece']['mean']:.4f} & {d['rare_recall']['mean']:.4f} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "main_results.tex").write_text("\n".join(lines) + "\n")

    # ---- extended data (operator / conformal / active / persistence) ----- #
    op = summary["operator"]; conf = summary["conformal"]["std_inductive"]
    act = summary["active_sampling"]; pers = summary["persistence"]
    h = summary["headline"]; tgt = summary["conformal"]["target_coverage"]

    def pm(d):
        return f"{d['mean']:.4f} $\\pm$ {d['ci95']:.4f}"

    ext = [r"\begin{tabular}{lc}", r"\toprule", r"quantity & value \\", r"\midrule",
           f"Directed-propagator non-normality $\\nu(P)$ & {pm(op['nonnormality_directed'])} \\\\",
           f"Undirected-walk non-normality $\\nu(P_0)$ & {pm(op['nonnormality_undirected'])} \\\\",
           f"Truncation commutator $\\|[M_\\phi,M_\\psi]\\|_F$ & {op['truncation_commutator_norm']['mean']:.4f} \\\\",
           f"Operator-kernel min.\\ eigenvalue (PD witness) & {op['kernel_min_eigenvalue']['mean']:.4f} \\\\",
           f"Diffusion spectral radius $\\rho(\\alpha B)$ & {op['diffusion_spectral_radius']['mean']:.4f} \\\\",
           r"\midrule",
           f"Conformal target coverage $1-\\delta$ & {tgt:.2f} \\\\",
           f"Conformal marginal coverage & {pm(conf['coverage'])} \\\\",
           f"Conformal rare-state coverage & {pm(conf['rare_coverage'])} \\\\",
           f"Average prediction-set size & {conf['avg_set_size']['mean']:.4f} \\\\",
           f"ECE, raw $\\to$ temperature-scaled & {conf['ece_raw']['mean']:.4f} $\\to$ {conf['ece_calibrated']['mean']:.4f} \\\\",
           r"\midrule",
           f"Active sampling, persistence rare recall & {pm(act['persistence']['rare_recall'])} \\\\",
           f"Active sampling, density rare recall & {pm(act['topology']['rare_recall'])} \\\\",
           f"Active sampling, random rare recall & {pm(act['random']['rare_recall'])} \\\\",
           f"Persistence-policy rare-query rate & {pm(act['persistence']['rare_query_rate'])} \\\\",
           r"\midrule",
           f"$H_0$ max persistence & {pers['max_persistence']['mean']:.4f} \\\\",
           f"$H_0$ persistence entropy & {pers['persistence_entropy']['mean']:.4f} \\\\",
           r"\bottomrule", r"\end{tabular}"]
    (out / "extended_results.tex").write_text("\n".join(ext) + "\n")

    print("wrote results/main_results.tex and results/extended_results.tex")
    print(f"  inductive STDD acc {h['std_inductive_accuracy']:.4f} "
          f"(+{h['inductive_accuracy_gain']:.4f} vs kNN); "
          f"rare recall {h['std_inductive_rare_recall']:.4f} vs {h['baseline_rare_recall']:.4f}")


if __name__ == "__main__":
    main()
