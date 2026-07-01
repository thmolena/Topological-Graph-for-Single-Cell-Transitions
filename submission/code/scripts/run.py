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
          f"inductive STDD acc={h['std_inductive_accuracy']:.4f} "
          f"(+{h['inductive_accuracy_gain']:.4f} vs kNN)  "
          f"rare-recall={h['std_inductive_rare_recall']:.4f} "
          f"(vs {h['baseline_rare_recall']:.4f} kNN)  "
          f"conformal cov={h['conformal_coverage']:.3f}/{h['conformal_target']:.2f}  "
          f"ECE {h['ece_raw']:.3f}->{h['ece_calibrated']:.3f}  "
          f"nonnorm={h['nonnormality_directed']:.3f}  "
          f"runtime={summary['provenance']['runtime_sec']}s")


if __name__ == "__main__":
    main()
