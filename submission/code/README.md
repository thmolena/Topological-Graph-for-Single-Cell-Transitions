# topocell — code artifact

CPU reference implementation for **Topological Graph RL for Single-Cell State
Transition and Perturbation Forecasting**. The benchmark is a seeded synthetic
**branching lineage** (cells progressing along a tree of fates over pseudotime,
with a deliberately rare state); forecasting is done by **label propagation on a
kNN cell graph** (graph-smoothed) against a plain kNN classifier, and an
**active-sampling** policy is compared to random querying for rare-state recall.
Real data plugs in via honest stubs (`ingest.py`); nothing is fabricated.

## Install
```bash
conda env create -f environment.yml && conda activate topocell   # or:
pip install -r requirements.txt && pip install -e .
```

## Reproduce
```bash
make test        # synthetic structure · no-leakage splits · Betti-0 monotonicity · metric bounds
make demo        # smoke config (~1-4 s) -> results/summary.json
make tables      # results/main_results.{tex,md}
make figures     # figures/fig_lineage.pdf, figures/fig_forecast.pdf
make audit       # readiness gate
make full-run    # reported-scale config (a minute or two)
# or, one command:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```
> macOS: the Makefile/scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-PyTorch can both ship an OpenMP runtime. No effect on results.

## Layout
```
src/topocell/
  synthetic.py   seeded branching lineage (branches · states · pseudotime ·
                 donor/batch/perturbation · a rare state)
  ingest.py      honest STUB hooks for real AnnData / perturbation screens
  graph.py       kNN cell graph · Betti-0 (component count) vs k · density
  splits.py      donor / batch / time / perturbation splits + leakage check
  policy.py      graph-smoothed label propagation vs kNN baseline;
                 topology-aware active sampling vs random
  metrics.py     accuracy · ECE · rare-state recall · summaries
  runner.py      end-to-end protocol -> results/summary.json
  audit.py       forbidden-claims + traceable-number checks
  plotting.py    lineage scatter + forecast/rare-recall bars
  seed.py        deterministic seeding + run provenance
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     synthetic structure · no-leakage splits · Betti-0 monotonicity · metric bounds
```

## What is computed
`results/summary.json` holds, per split protocol: forecast accuracy, ECE and
rare-state recall for the graph-smoothed method and the baseline; the
active-vs-random rare-state-recall comparison; the Betti-0 curve over k; and a
`headline` dict with the graph-smoothed accuracy, its gain over the baseline,
and the rare-state-recall gain from active sampling. Integrity flags
(`donor_split_clean`, `betti_monotonic`) gate the audit. It is the single source
of truth for every table, figure and macro.

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
