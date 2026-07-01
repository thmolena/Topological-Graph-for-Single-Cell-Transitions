# specops-dsto

Reference implementation for **"Directed spectral-truncated operators for
inductive classification of quantum phases across a control-parameter sweep"**
(Molena Huynh, 2026; BibTeX key `huynh2026dsto`).

This is the fifteenth entry in the **spectral-truncation-operators** research
program, which treats the spectral truncation of operators as a common design
primitive across quantum optimization, simulation, error correction, and phase
classification. Here that primitive is carried onto the *state graph* of a
parameterized quantum many-body system: a non-self-adjoint, drive-directed
propagator transports phase labels along a control-parameter sweep; a
band-limiting spectral truncation denoises it into a closed-form solve and
induces a noncommutative operator `*`-algebra with a positive-definite kernel;
an out-of-sample Nystrom extension makes the classifier inductive; and a
split-conformal layer supplies distribution-free coverage for the rare critical
regime.

Everything is CPU-only numpy/scipy/scikit-learn and fully deterministic.
All experiments are reproducible on commodity hardware; runtime and memory are reported for each benchmark.

## Install

```bash
pip install .
```

Editable / development install (adds `pytest`):

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.9. The install exposes two identical console entry points,
`dsto-reproduce` and `topocell-reproduce`, and importable package `topocell`.

## Reproduce the paper

Regenerate every figure and table from a fresh run of the pipeline:

```bash
dsto-reproduce                       # runs configs/full.yaml, writes figures + tables
dsto-reproduce --config configs/smoke.yaml   # fast laptop-scale demo
dsto-reproduce --skip-run            # rebuild figures/tables from existing summary.json
```

Or drive the stages directly:

```bash
python scripts/run.py --config configs/full.yaml --out results   # -> results/summary.json
python scripts/make_tables.py                                     # -> results/*.tex
python scripts/make_figures.py                                    # -> figures/*.pdf
python scripts/audit_claims.py                                    # readiness gate
make demo        # smoke run + tables + figures
make full        # full-scale run + tables + figures
make audit       # run the claim-audit gate
make test        # pytest
```

`results/summary.json` is the single source of truth: every table, figure, and
manuscript number is derived from it, and `scripts/audit_claims.py` fails if any
headline number is not traceable to a real run. The full-scale configuration
(`n_cells=3000`, `n_seeds=5`) runs in about 45 s on a laptop CPU.

## Cite this work

```bibtex
@article{huynh2026dsto,
  author  = {Huynh, Molena},
  title   = {Directed spectral-truncated operators for inductive classification
             of quantum phases across a control-parameter sweep},
  year    = {2026},
  note    = {Part of the spectral-truncation-operators program},
}
```

## Extend / tweak

All knobs live in a single YAML config parsed into `topocell.config.Config`
(`src/topocell/config.py`). Pass any config to `--config`; unspecified fields
fall back to the dataclass defaults. `configs/smoke.yaml` is the fast demo and
`configs/full.yaml` is the reported-scale run.

### Tunable parameters (config fields / YAML keys)

| Key | Meaning | Module |
| --- | --- | --- |
| `name` | run label, stamped into `summary.json` | config |
| `seed` | base RNG seed; seed `s` uses `seed + s` | seed |
| `n_cells` | number of quantum-state samples | synthetic |
| `n_features` | shadow-feature (classical-shadow) dimension | synthetic |
| `n_branches` | number of ordered phases / lineage branches | synthetic |
| `n_donors` | number of synthetic devices (donor split) | synthetic/splits |
| `n_batches` | number of shot-noise batches (batch split) | synthetic/splits |
| `n_perturbations` | number of Hamiltonian perturbations (perturbation split) | synthetic/splits |
| `noise` | shadow-feature noise scale | synthetic |
| `rare_fraction` | target fraction of the rare critical regime | synthetic |
| `k` | neighborhood size of the kNN state graph | graph |
| `label_fraction` | few-label annotation budget (stratified for the rare state) | splits |
| `splits` | which leakage-checked protocols to run: `donor,batch,time,perturbation` | splits |
| `active_budget` | active-sampling query budget | policy |
| `n_seeds` | number of seeds aggregated behind each CI | runner |
| `rank` | spectral-truncation level `r` (band-limit) | operators |
| `alpha` | diffusion / clamping trade-off; must keep `rho(alpha B) < 1` | operators |
| `beta` | pseudotime-direction strength (`0` = undirected control) | operators |
| `eps` | directed-walk laziness in `(0, 1]` | operators |
| `tau` | pseudotime scale of the directional kernel | operators |
| `conformal_delta` | target miscoverage (coverage >= `1 - delta`) | conformal |
| `cal_fraction` | fraction of annotated cells held for calibration | conformal |

### CLI flags

- `scripts/run.py --config PATH --out DIR`
- `scripts/make_tables.py`, `scripts/make_figures.py` (read `results/summary.json`)
- `dsto-reproduce [--config PATH] [--skip-run]`

### Adding a new parameter

1. Add a field (with a default) to the `Config` dataclass in
   `src/topocell/config.py`; YAML loading picks it up automatically.
2. Thread it into the relevant module (`operators.py`, `synthetic.py`,
   `conformal.py`, `policy.py`, `splits.py`).
3. If it should appear in the paper, surface it in the `summary["headline"]`
   dict in `src/topocell/runner.py` and reference it from
   `scripts/make_tables.py` / `scripts/make_figures.py`.

### Adding a new input / dataset

Replace `topocell.synthetic.make_lineage` (or add an ingestor in
`topocell.ingest`) that returns the same `Lineage` structure: a cell-by-feature
matrix `X`, integer `state` labels, a scalar `pseudotime`/control parameter per
cell, and a designated `rare_state`. Everything downstream (graph, operators,
splits, conformal, persistence) is dataset-agnostic and consumes only that
interface.

### Adding a new forecaster / operator

Implement a class exposing `predict(...) -> Forecast` (see
`topocell.policy.Forecast`), register its name in `FORECASTERS` and the
`_forecast` dispatch in `src/topocell/runner.py`, and it will be scored under
every split automatically.

### Plugging into other projects

Import the operator directly:

```python
from topocell.operators import InductiveSpectralTruncatedDiffusion
fc = InductiveSpectralTruncatedDiffusion(k=20, rank=80, alpha=0.9,
                                         beta=1.0, eps=0.6, tau=0.2)
forecast = fc.predict(X, state, train_mask, test_mask, labeled_mask, pseudotime)
```

The conformal layer (`topocell.conformal`) and the noncommutativity /
truncation diagnostics (`topocell.operators.nonnormality`,
`truncation_commutator_norm`) are reusable on any graph-structured problem.
