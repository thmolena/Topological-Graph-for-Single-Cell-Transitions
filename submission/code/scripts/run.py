#!/usr/bin/env python
"""Run the single-cell forecasting experiment and write results/summary.json."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from topocell.config import Config
from topocell.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    summary = run(cfg, Path(args.out))
    h = summary["headline"]
    print(f"[{cfg.name}] cells={h['n_cells']}  states={h['n_states']}  "
          f"graph-smoothed acc={h['graph_smoothed_accuracy']:.4f}  "
          f"(+{h['accuracy_gain_vs_baseline']:.4f} vs baseline)  "
          f"active rare-recall={h['active_rare_recall']:.4f} "
          f"(+{h['active_rare_recall_gain']:.4f} vs random)  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
