# topocell — code artifact

CPU reference implementation for topological graph learning of single-cell state
transitions. Held-out cell states are forecast by label propagation on a symmetric
k-nearest-neighbour cell graph (graph-smoothed), evaluated against a non-graph kNN classifier;
a topology-aware active-sampling policy (uncertainty weighted by inverse local density) is
evaluated against random querying for rare-state recall. The benchmark is a seeded synthetic
branching lineage carrying donor, batch, pseudotime, and perturbation metadata; transfer is
measured under four leakage-controlled held-out split protocols. Measured data attaches through
`ingest.py` stubs.

## Installation

```bash
pip install topocell                # from PyPI, once published

# from source (this directory)
conda env create -f environment.yml && conda activate topocell
pip install -e .                    # installs the topocell package (CPU reference)
pip install -e .[gpu]               # optional: torch GPU/GNN backend
```

The scripts also place `src/` on `sys.path`, so an editable install is optional;
`export PYTHONPATH=src` suffices to run the reproduction targets from this directory.

## Reproduce

The installed console entry point regenerates the results artifact, tables, and figures from a
deterministic seeded run:

```bash
topocell-reproduce                              # reported scale (configs/full.yaml)
topocell-reproduce --config configs/smoke.yaml  # laptop-scale smoke run
topocell-reproduce --skip-run                   # tables and figures from existing summary.json
```

The Makefile exposes the same pipeline step by step:

```bash
make test        # synthetic structure · leakage-free splits · Betti-0 monotonicity · metric bounds
make demo        # smoke config (configs/smoke.yaml) -> results/summary.json
make full-run    # reported-scale config (configs/full.yaml) -> results/summary.json
make tables      # results/main_results.tex, results/main_results.md
make figures     # figures/fig_lineage.pdf, figures/fig_forecast.pdf
make audit       # readiness gate: traceable numbers, integrity flags, no forbidden claims
# or, one command end to end:
bash scripts/reproduce_all.sh         # smoke
bash scripts/reproduce_all.sh full    # reported scale
```

> macOS: the Makefile and scripts set `KMP_DUPLICATE_LIB_OK=TRUE` because conda and
> pip-installed PyTorch can each bundle an OpenMP runtime. This has no effect on results.

## Figures and tables regenerated

| Artifact | Producer | Contents |
|---|---|---|
| `results/summary.json` | `scripts/run.py` | Per-split accuracy, ECE, and rare-state recall for the graph-smoothed method and the baseline; the active-versus-random rare-recall comparison; the Betti-0 curve over `k`; integrity flags; and the headline block. Authoritative record for every table, figure, and macro. |
| `results/main_results.tex` | `scripts/make_tables.py` | LaTeX per-split results table. |
| `results/main_results.md` | `scripts/make_tables.py` | Markdown per-split results table (transcribed into the project README). |
| `figures/fig_lineage.pdf` | `scripts/make_figures.py` | Lineage scatter of the synthetic branching benchmark. |
| `figures/fig_forecast.pdf` | `scripts/make_figures.py` | Forecast accuracy and rare-state-recall bars (method versus baseline). |

## Determinism (seeds)

Runs are seeded end to end. Each configuration fixes a base `seed` (0 in both `smoke.yaml` and
`full.yaml`); `runner.py` derives one seed per repetition as `seed + s` for `s` in
`0 .. n_seeds - 1`, and seeds the synthetic lineage, the split assignment, and the
active-sampling policy from it. `topocell.seed.set_seed` fixes the global NumPy state and
records run provenance (interpreter and library versions, platform, wall-clock runtime, and
peak memory) into `results/summary.json`. The reproduction entry point pins
`OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` for cross-machine stability. Two integrity
flags gate the audit: `donor_split_clean` (the held-out group is never annotated) and
`betti_monotonic` (the component count is monotone non-increasing in `k`).

## Pinned dependencies

Runtime dependencies are declared with bounded ranges in `pyproject.toml`:

```
numpy>=1.24,<3      scipy>=1.10,<2      networkx>=3.0,<4
scikit-learn>=1.2,<2  matplotlib>=3.6,<4  pyyaml>=6.0,<7
```

Optional extras: `gpu` (`torch>=2.0,<3`) and `dev` (`pytest`, `build`, `twine`). The exact
interpreter and library versions used for the committed `results/summary.json` are recorded in
its `provenance` block (Python 3.13.12, NumPy 2.4.3, SciPy 1.17.1, networkx 3.6.1,
scikit-learn 1.8.0, matplotlib 3.10.8).

## Layout

```
src/topocell/
  synthetic.py   seeded branching lineage (branches · states · pseudotime ·
                 donor/batch/perturbation · a rare state)
  ingest.py      stub hooks for AnnData / perturbation screens
  graph.py       kNN cell graph · Betti-0 (component count) over k · density
  splits.py      donor / batch / time / perturbation splits + leakage check
  policy.py      graph-smoothed label propagation vs kNN baseline;
                 topology-aware active sampling vs random
  metrics.py     accuracy · ECE · rare-state recall · summaries
  runner.py      end-to-end protocol -> results/summary.json
  audit.py       forbidden-claim + traceable-number checks
  plotting.py    lineage scatter + forecast/rare-recall bars
  reproduce.py   installed reproduction command (topocell-reproduce)
  config.py · seed.py   YAML config loading · deterministic seeding + provenance
scripts/   run.py · make_tables.py · make_figures.py · audit_claims.py · reproduce_all.sh
configs/   smoke.yaml (demo) · full.yaml (reported)
tests/     synthetic structure · leakage-free splits · Betti-0 monotonicity · metric bounds
```

All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.
