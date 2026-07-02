# specops-dsto

Reference implementation for **"Directed spectral-truncated operators for
inductive classification of quantum phases across a control-parameter sweep"**
(Molena Huynh, North Carolina State University, 2026; BibTeX key
`huynh2026dsto`).

## Summary

A programmable quantum simulator prepares and measures a many-body state at each
setting of a control parameter that is swept across a phase transition. Reading
that device is statistical inference: each preparation returns one measurement
record, so identifying the phase a state occupies is estimation of a discrete
label from finite, noisy samples, and it must carry calibrated uncertainty. This
package implements a single non-self-adjoint graph operator that performs that
inference. It builds a directed state graph over classical-shadow feature
vectors, transports phase labels forward along the sweep with a drive-directed
propagator, band-limits and denoises the diffusion by spectral truncation into a
closed-form solve, extends the classifier to unseen states by an out-of-sample
Nystrom step, and calibrates the rare-regime prediction with a split-conformal
layer. Everything is CPU-only NumPy/SciPy/scikit-learn, fully seeded, and
deterministic; every table, figure, and manuscript number is regenerated from a
single run artifact `results/summary.json`.

## Background and problem setting

A quantum simulator is, from the keyboard, a stochastic device interrogated by
sampling. The Born rule means each run returns a single bitstring drawn from a
distribution, so recognizing the *phase* a prepared state occupies is not a
deterministic readout but an inference problem: estimate a discrete label from a
finite, noisy sample and attach honest uncertainty to it. The standard feature
representation is the *classical shadow* — a randomized-measurement statistic
that compresses an exponentially large state into a tractable, low-dimensional
vector. Across a control-parameter sweep these shadow features concentrate near
a low-dimensional manifold (the phase diagram), routinely approximated by a
*k*-nearest-neighbor (*k*NN) similarity graph.

The canonical estimator on such a graph is graph-based semi-supervised learning:
phase labels are diffused over the state graph so that each state borrows
statistical strength from its neighborhood. The diffusion operator used almost
universally is the *symmetric* normalized adjacency `S = D^{-1/2} A D^{-1/2}`.
Two structural defects limit it for classification across a sweep.

1. **It is self-adjoint, hence reversible.** It cannot encode the directed arrow
   of the drive, even though the control-parameter sweep supplies exactly that
   arrow and directed, non-Hermitian transition operators are the native
   language of driven and open quantum dynamics.
2. **Built over all states, it is transductive.** It consults the held-out
   states' connectivity at inference, so a measured accuracy gain over an
   inductive point classifier conflates the value of diffusion with a
   transductive information advantage — the decisive missing control in
   comparable studies.

A second difficulty is that the physically decisive regimes — quantum critical
points, narrow topological windows — are rare, and classifiers trained to
minimize average error miss them. This implementation turns both operator
defects and the rare-regime problem into design targets.

## Contributions

1. **A drive-directed, noncommutative graph propagator.** Reweighting each *k*NN
   edge `i → j` by a bounded forward kernel `κ(s_j − s_i) = exp(β tanh((s_j −
   s_i)/τ))` of the control-parameter increment and row-normalizing yields a
   non-self-adjoint operator `P = D_W^{-1} W`. Its relative non-normality
   `ν(P) = ‖PPᵀ − PᵀP‖_F / ‖P‖_F²` is strictly positive (ν(P) = 0.0370) and the
   drive contributes ~36% of it over the undirected control (ν(P₀) = 0.0272).
   The directed transport carries labels *along* the sweep rather than merely
   smoothing them across it.
2. **Spectral truncation with proved optimality.** Band-limiting the diffusion
   to the leading `r` graph-Fourier modes `Π_r = U_r U_rᵀ` is the
   Eckart–Young–Mirsky-optimal rank-`r` low-pass approximation of the monotone
   positive-semidefinite diffusion filter, and it compresses the solve to `r`
   coordinates. The truncated multiplication operators
   `M_φ^{(r)} = U_rᵀ diag(φ) U_r` do not commute — the spectral-truncation
   noncommutativity of noncommutative geometry, here on the state graph — and
   they generate a positive-definite Gram kernel `K = Σ_c M_c M_cᵀ`
   (λ_min = 1.62 > 0).
3. **An inductive operator that removes the transductive confound.** An
   out-of-sample Nystrom/Nadaraya–Watson extension labels a held-out state from
   the training field alone, forming no test–test edge and reading no held-out
   adjacency. The matched inductive control is therefore the method itself, and
   the trained operator transfers to states from a device it never saw.
4. **Rare-regime recovery and forward extrapolation.** On the four
   leakage-checked transfer protocols the inductive operator beats a matched
   inductive *k*NN baseline in mean accuracy (0.6856 vs 0.5992), lifts mean
   rare-state recall to 0.3322 from a near-zero baseline (0.0151), and raises
   forward-extrapolation (schedule-split) accuracy by 0.1749 over the point
   classifier.
5. **Distribution-free conformal calibration.** A split-conformal layer attains
   realized marginal coverage 0.8968 against a 0.90 target on an exchangeable
   hold-out; temperature scaling cuts expected calibration error from 0.1374 to
   0.0369 (73% reduction).
6. **A genuine multiscale topology signal.** The degree-0 persistence barcode,
   computed exactly by union–find on the *k*NN distance graph, replaces the
   passive connected-component count of prior work; a persistence-guided active
   sampler raises rare-state recall to 0.3941 against 0.0481 for random querying.
7. **Operator-theoretic guarantees and deterministic reproducibility.** The
   accompanying manuscript proves truncation optimality, non-normality,
   well-posedness, inductive consistency, and conformal coverage; this package
   regenerates every table, figure, and number from one seeded `summary.json`.

## Method

Fix `n` states embedded in a shadow-feature space with a symmetric *k*NN
adjacency `A`, degrees `D = diag(d_i)`, and a control-parameter coordinate
`s ∈ ℝⁿ`. The classifier is a single operator assembled in three stages.

- **Directed propagator.** Each directed edge is reweighted by the bounded
  forward kernel `W_ij = A_ij κ(s_j − s_i)` and row-normalized to a transition
  matrix `P = D_W^{-1} W`. Forward edges (`s_j > s_i`) are up-weighted and
  backward edges down-weighted; the `tanh` keeps the reweighting bounded. `P` is
  non-self-adjoint, so it transports labels along the sweep.
- **Spectral truncation.** With `U_r` the eigenvectors of `S` for its `r`
  largest eigenvalues (the smoothest graph-Fourier modes) and
  `Π_r = U_r U_rᵀ`, the lazy directed walk `P̃ = (1 − ε)I + εP` is compressed
  to `B = U_rᵀ P̃ U_r ∈ ℝ^{r×r}`. Seeding one-hot phase labels `F₀` on the
  annotated states, the band-limited field is solved in closed form,
  `F★ = U_r (I − αB)^{-1} (1 − α) U_rᵀ F₀`.
- **Inductive extension and calibration.** `A`, `U_r`, and `P` are built on the
  training states only; a held-out state `x*` is labeled by a Gaussian-weighted
  average of `F★` over its nearest *training* states. A split-conformal layer
  computes nonconformity scores `1 − p_y` on a held-out calibration slice and
  thresholds at the conformal quantile for finite-sample coverage `≥ 1 − δ`
  under exchangeability; temperature scaling recalibrates the scalar confidence.

The classifier compared throughout is the inductive spectral-truncated
directed-diffusion operator (**inductive STDD**), against its transductive
variant, symmetric label propagation, and a matched inductive *k*NN baseline.

## Main results

All numbers derive from a fully seeded synthetic benchmark that emulates the
shadow-feature geometry of a parameterized quantum system (three phases — a
disordered parent phase and two ordered phases bifurcating at the critical
control parameter `s₀ = 0.45` — with a deliberately rare critical regime at ~4%
of states). The full-scale configuration generates 3000 states with 16 shadow
features over 5 seeds; the run completes in ~32 s at ~621 MB peak memory.

- **Held-out accuracy on every protocol.** Inductive STDD vs matched *k*NN:
  device 0.7227 vs 0.6550, shot-noise 0.7134 vs 0.6355, Hamiltonian 0.6741 vs
  0.6490, drive-schedule 0.6323 vs 0.4574. Mean 0.6856 vs 0.5992 (gain 0.0864);
  it improves on *every* split, and even exceeds transductive symmetric label
  propagation (0.6587) despite operating in the harder inductive regime.
- **Rare-regime recovery and forward rescue.** Mean rare-state recall 0.3322 vs
  0.0151 for the point classifier. The largest accuracy margin is on the
  forward-extrapolation drive-schedule split (+0.1749 over *k*NN, +0.0850 over
  symmetric label propagation). Rare-state recall on that split is 0.0000 for
  all methods — the rare states lie beyond the labeled control-parameter range,
  a null reported as is.
- **Operator structure.** ν(P) = 0.0370 vs ν(P₀) = 0.0272 (undirected);
  truncation commutator ‖[M_φ, M_ψ]‖_F = 5.93; Gram-kernel λ_min = 1.62 > 0;
  spectral radius ρ(αB) = 0.9209 < 1 at the fixed rank `r = 80`.
- **Calibration.** Realized conformal coverage 0.8968 vs 0.90 target (rare-regime
  coverage 0.9307, mean set size 1.81 of four phases); ECE 0.1374 → 0.0369.
- **Active sampling.** Persistence-guided querying reaches rare-state recall
  0.3941 (± 0.1910) vs 0.3556 (inverse-density) and 0.0481 (random), spending
  40% of its budget on genuinely rare states against a 4% base rate.

## Significance

The construction connects graph-based semi-supervised learning to the
machine-learning-for-quantum-physics program of recognizing phases from
measurement data. Its methodological advance is to build the classifier from a
spectrally truncated, drive-directed operator — instantiating, on the state
graph, the spectral truncation and operator noncommutativity of `C*`-algebraic
kernel machines — and to make it inductive and conformally calibrated. The
inductive extension removes the transductive information advantage that inflates
symmetric-smoothing studies, so the accuracy gain is attributable to the
operator rather than to a confounded design choice, and the trained operator
transfers to states from an unseen device. All data here are synthetic; the
decisive test is validation on classical shadows from a programmable quantum
simulator, for which the repository ships honest, unexercised ingest stubs.

## Installation and reproduction

```bash
pip install .
```

Editable / development install (adds `pytest`):

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.9; CPU-only (NumPy, SciPy, scikit-learn, NetworkX,
Matplotlib). The install exposes two identical console entry points,
`dsto-reproduce` and `topocell-reproduce`, and the importable package
`topocell`.

Regenerate every figure and table from a fresh run of the pipeline:

```bash
dsto-reproduce                                 # runs configs/full.yaml, writes figures + tables
dsto-reproduce --config configs/smoke.yaml     # fast laptop-scale demo (few hundred states, one seed)
dsto-reproduce --skip-run                       # rebuild figures/tables from existing summary.json
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
headline number is not traceable to a real run. Three integrity gates must pass
before any metric is recorded: the grouping splits are leakage-free, the rare
regime is present, and the directed propagator is non-normal. The full-scale
configuration (`n_cells=3000`, `n_seeds=5`) runs in about 32 s on a laptop CPU.

## Extend / tweak

All knobs live in a single YAML config parsed into `topocell.config.Config`
(`src/topocell/config.py`). Pass any config to `--config`; unspecified fields
fall back to the dataclass defaults. `configs/smoke.yaml` is the fast demo and
`configs/full.yaml` is the reported-scale run.

### Tunable parameters (config fields / YAML keys)

The config field names carry a lineage-generator vocabulary from an earlier
version of the code; the quantum-phase meaning each field takes in this work is
given in the second column.

| Key | Meaning in this work | Module |
| --- | --- | --- |
| `name` | run label, stamped into `summary.json` | config |
| `seed` | base RNG seed; seed `s` uses `seed + s` | seed |
| `n_cells` | number of quantum-state samples | synthetic |
| `n_features` | shadow-feature (classical-shadow) dimension | synthetic |
| `n_branches` | number of phases (one disordered + ordered branches) | synthetic |
| `n_donors` | number of synthetic devices (device split) | synthetic/splits |
| `n_batches` | number of shot-noise batches (shot-noise split) | synthetic/splits |
| `n_perturbations` | number of Hamiltonian perturbations (Hamiltonian split) | synthetic/splits |
| `noise` | shadow-feature noise scale | synthetic |
| `rare_fraction` | target fraction of the rare critical regime | synthetic |
| `k` | neighborhood size of the *k*NN state graph | graph |
| `label_fraction` | few-label annotation budget (stratified for the rare regime) | splits |
| `splits` | leakage-checked protocols: `donor` (device), `batch` (shot-noise), `time` (drive schedule), `perturbation` (Hamiltonian) | splits |
| `active_budget` | active-sampling query budget | policy |
| `n_seeds` | number of seeds aggregated behind each CI | runner |
| `rank` | spectral-truncation level `r` (band-limit) | operators |
| `alpha` | diffusion / clamping trade-off; must keep `ρ(αB) < 1` | operators |
| `beta` | drive-direction strength (`0` = undirected control) | operators |
| `eps` | directed-walk laziness in `(0, 1]` | operators |
| `tau` | control-parameter scale of the directional kernel | operators |
| `conformal_delta` | target miscoverage (coverage >= `1 − delta`) | conformal |
| `cal_fraction` | fraction of annotated states held for calibration | conformal |

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

Provide a generator (or an ingestor in `topocell.ingest`) that returns the same
structure the synthetic benchmark does: a state-by-feature matrix `X`, integer
phase `state` labels, a scalar control parameter per state, and a designated
`rare_state`. Everything downstream (graph, operators, splits, conformal,
persistence) is dataset-agnostic and consumes only that interface. Honest ingest
stubs for real classical-shadow and perturbed-Hamiltonian data are included but
intentionally not exercised.

### Adding a new classifier / operator

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
