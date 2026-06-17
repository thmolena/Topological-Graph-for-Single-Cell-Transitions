#!/usr/bin/env python
"""Generate LaTeX/Markdown tables from results/summary.json.

The main table is per-split: forecast accuracy, ECE and rare-state recall for
the graph-smoothed method vs the non-graph baseline.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    tables = Path("results")
    tables.mkdir(exist_ok=True)
    by_split = summary["by_split"]

    rows = []  # (split, method, acc, ece, rare_recall)
    for split, m in by_split.items():
        for method, label in (("graph_smoothed", "graph-smoothed"),
                              ("baseline_knn", "baseline kNN")):
            rows.append((split, label,
                         m[method]["accuracy"]["mean"],
                         m[method]["ece"]["mean"],
                         m[method]["rare_recall"]["mean"]))

    # LaTeX
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"split & method & accuracy & ECE & rare-state recall \\",
             r"\midrule"]
    for split, method, acc, ece, rr in rows:
        lines.append(f"{split} & {method} & {acc:.4f} & {ece:.4f} & {rr:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(lines) + "\n")

    # Markdown (for the README)
    md = ["| split | method | accuracy | ECE | rare-state recall |",
          "|---|---|---|---|---|"]
    for split, method, acc, ece, rr in rows:
        bold = "**" if method == "graph-smoothed" else ""
        md.append(f"| {split} | {bold}{method}{bold} | {bold}{acc:.4f}{bold} | {ece:.4f} | {bold}{rr:.4f}{bold} |")
    (tables / "main_results.md").write_text("\n".join(md) + "\n")
    print("wrote results/main_results.tex, results/main_results.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
