"""End-to-end experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. For each random seed and each leakage-checked split protocol
(donor / batch / time / perturbation) the runner evaluates four forecasters on
an identical small annotated set:

  * ``baseline_knn``      -- a plain (inductive) kNN point classifier;
  * ``label_prop``        -- transductive symmetric label propagation (the prior
                            graph-smoothing baseline);
  * ``std_transductive``  -- spectral-truncated *directed* diffusion, transductive;
  * ``std_inductive``     -- the same operator made *inductive* by an
                            out-of-sample Nystrom extension (the headline method,
                            which never sees held-out connectivity).

It then (i) calibrates the inductive forecaster by split-conformal prediction and
temperature scaling on an exchangeable hold-out, reporting realized coverage and
the calibrated-vs-raw expected calibration error; (ii) records operator
diagnostics -- the relative non-normality of the directed propagator (its
noncommutativity witness) against the undirected control, and a truncated-
multiplication commutator norm; (iii) computes the degree-0 persistence barcode
of the cell cloud and its Betti-0(scale) curve; and (iv) compares random,
inverse-density and persistence-based active sampling for rare-state recovery.
Integrity gates assert the splits are leakage-free, the directed propagator is
genuinely non-normal, and the rare state is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from . import metrics
from .config import Config
from .conformal import (
    apply_temperature,
    conformal_threshold,
    empirical_coverage,
    fit_temperature,
    prediction_sets,
)
from .operators import (
    InductiveSpectralTruncatedDiffusion,
    SpectralTruncatedDiffusion,
    directed_propagator,
    graph_fourier_basis,
    kernel_min_eigenvalue,
    nonnormality,
    spectral_radius,
    truncated_directed_operator,
    truncated_operator_kernel,
    truncation_commutator_norm,
)
from .persistence import h0_barcode, per_cell_isolation, persistence_features
from .policy import (
    KNNForecaster,
    LabelPropagationForecaster,
    active_sample,
    recall_after_query,
)
from .graph import build_knn_graph
from .seed import RunProvenance, set_seed
from .splits import all_splits, has_leakage
from .synthetic import make_lineage

FORECASTERS = ["baseline_knn", "label_prop", "std_transductive", "std_inductive"]


def _forecast(name, lin, split, cfg):
    """Run one named forecaster on one split and return its Forecast."""
    if name == "baseline_knn":
        return KNNForecaster(k=cfg.k).predict(lin.X, lin.state, split.labeled, split.test)
    if name == "label_prop":
        return LabelPropagationForecaster(k=cfg.k).predict(
            lin.X, lin.state, split.labeled, split.test)
    if name == "std_transductive":
        return SpectralTruncatedDiffusion(
            k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(lin.X, lin.state, split.labeled, split.test, lin.pseudotime)
    if name == "std_inductive":
        return InductiveSpectralTruncatedDiffusion(
            k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(lin.X, lin.state, split.train, split.test,
                                 split.labeled, lin.pseudotime)
    raise ValueError(name)


def _eval_forecasters(lin, split, cfg) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    y_test = lin.state[split.test]
    for name in FORECASTERS:
        fc = _forecast(name, lin, split, cfg)
        out[name] = {
            "accuracy": metrics.accuracy(y_test, fc.pred),
            "ece": metrics.expected_calibration_error(y_test, fc.pred, fc.confidence),
            "rare_recall": metrics.rare_state_recall(y_test, fc.pred, lin.rare_state),
        }
    return out


def _conformal_eval(lin, cfg, rng) -> Dict[str, float]:
    """Conformal + temperature calibration of the inductive forecaster.

    Uses an *exchangeable* random hold-out (so the split-conformal coverage
    guarantee applies): cells are split into train / calibration / test
    uniformly at random; the inductive forecaster is fit on the annotated train
    cells, calibrated on the annotated calibration cells, and the realized
    coverage, set size and raw-vs-calibrated ECE are measured on the test cells.
    """
    n = lin.n_cells
    perm = rng.permutation(n)
    n_test = n // 4
    n_cal = n // 4
    test_idx = perm[:n_test]
    cal_idx = perm[n_test:n_test + n_cal]
    train_idx = perm[n_test + n_cal:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    cal_mask = np.zeros(n, bool); cal_mask[cal_idx] = True
    test_mask = np.zeros(n, bool); test_mask[test_idx] = True

    # Annotate a small fraction of the train pool (stratified for the rare state).
    from .splits import _annotate
    labeled = _annotate(train_mask, lin, cfg.label_fraction, int(rng.integers(1 << 30)))

    model = InductiveSpectralTruncatedDiffusion(
        k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps, tau=cfg.tau)
    fc_cal = model.predict(lin.X, lin.state, train_mask, cal_mask, labeled, lin.pseudotime)
    fc_test = model.predict(lin.X, lin.state, train_mask, test_mask, labeled, lin.pseudotime)
    classes = fc_test.classes
    col = {int(c): j for j, c in enumerate(classes)}

    y_cal = lin.state[cal_mask]
    y_test = lin.state[test_mask]
    keep_cal = np.array([int(c) in col for c in y_cal])
    y_cal_idx = np.array([col[int(c)] for c in y_cal[keep_cal]])
    proba_cal = fc_cal.proba[keep_cal]

    T = fit_temperature(proba_cal, y_cal_idx)
    thr = conformal_threshold(proba_cal, y_cal_idx, delta=cfg.conformal_delta)
    sets = prediction_sets(fc_test.proba, thr)
    keep_test = np.array([int(c) in col for c in y_test])
    y_test_idx = np.array([col[int(c)] for c in y_test[keep_test]])
    rare_col = col.get(int(lin.rare_state))
    cov = empirical_coverage(sets[keep_test], y_test_idx, rare_col)

    ece_raw = metrics.expected_calibration_error(
        y_test[keep_test], classes[np.argmax(fc_test.proba[keep_test], 1)],
        fc_test.proba[keep_test].max(1))
    proba_ct = apply_temperature(fc_test.proba[keep_test], T)
    ece_cal = metrics.expected_calibration_error(
        y_test[keep_test], classes[np.argmax(proba_ct, 1)], proba_ct.max(1))
    return {
        "coverage": cov.coverage, "avg_set_size": cov.avg_set_size,
        "rare_coverage": cov.rare_coverage, "temperature": T,
        "ece_raw": ece_raw, "ece_calibrated": ece_cal,
    }


def _truncation_sweep(lin, cfg, split, ranks) -> Dict[str, List[float]]:
    """Inductive accuracy / rare-recall as a function of the truncation rank.

    Visualizes the spectral-truncation bias--variance trade-off: too small a rank
    under-fits, too large re-admits high-frequency noise. Computed on one split /
    seed for the operator figure.
    """
    accs, rrs = [], []
    y_test = lin.state[split.test]
    for r in ranks:
        fc = InductiveSpectralTruncatedDiffusion(
            k=cfg.k, rank=int(r), alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(lin.X, lin.state, split.train, split.test,
                                 split.labeled, lin.pseudotime)
        accs.append(metrics.accuracy(y_test, fc.pred))
        rrs.append(metrics.rare_state_recall(y_test, fc.pred, lin.rare_state))
    return {"ranks": [int(r) for r in ranks], "accuracy": accs, "rare_recall": rrs}


def _active_sampling_lift(lin, cfg, rng) -> Dict[str, Dict[str, float]]:
    n = lin.n_cells
    seed_mask = np.zeros(n, dtype=bool)
    seed_idx = rng.choice(n, size=max(10, n // 10), replace=False)
    seed_mask[seed_idx] = True
    results = {}
    for strategy in ("persistence", "topology", "random"):
        queried = active_sample(lin.X, lin.state, seed_mask, cfg.active_budget, k=cfg.k,
                                strategy=strategy,
                                rng=np.random.default_rng(int(rng.integers(1 << 30))))
        _, after = recall_after_query(lin.X, lin.state, seed_mask, queried,
                                      lin.rare_state, k=cfg.k)
        rare_hit = float(np.mean(lin.state[queried] == lin.rare_state)) if queried.size else 0.0
        results[strategy] = {"rare_recall_after": after, "rare_query_rate": rare_hit}
    return results


def run(cfg: Config, out_dir: Path) -> Dict:
    prov = RunProvenance(seed=cfg.seed)
    set_seed(cfg.seed)

    acc = {s: {f: [] for f in FORECASTERS} for s in cfg.splits}
    ece = {s: {f: [] for f in FORECASTERS} for s in cfg.splits}
    rare = {s: {f: [] for f in FORECASTERS} for s in cfg.splits}
    active = {s: {"rare_recall_after": [], "rare_query_rate": []}
              for s in ("persistence", "topology", "random")}
    conf_keys = ("coverage", "avg_set_size", "rare_coverage", "temperature",
                 "ece_raw", "ece_calibrated")
    conf = {key: [] for key in conf_keys}
    op_nn_dir, op_nn_undir, op_commutator = [], [], []
    op_kernel_mineig, op_spec_radius = [], []
    pf_keys = ("max_persistence", "total_persistence", "persistence_entropy", "n_significant")
    pers = {key: [] for key in pf_keys}
    splits_clean = True
    rare_fracs, n_states_seen = [], []
    betti_scales = list(np.round(np.linspace(1.0, 3.5, 6), 4))
    betti_curve_seed0: List[int] = []
    spectrum_seed0: List[float] = []
    trunc_sweep_seed0: Dict = {}
    sweep_ranks = [r for r in (8, 16, 32, 48, 64, 80, 120, 160, 240) if r < cfg.n_cells // 4]

    for s in range(cfg.n_seeds):
        seed = cfg.seed + s
        lin = make_lineage(
            n_cells=cfg.n_cells, n_features=cfg.n_features, n_branches=cfg.n_branches,
            n_donors=cfg.n_donors, n_batches=cfg.n_batches,
            n_perturbations=cfg.n_perturbations, noise=cfg.noise,
            rare_fraction=cfg.rare_fraction, seed=seed)
        rare_fracs.append(lin.state_frequencies().get(lin.rare_state, 0.0))
        n_states_seen.append(lin.n_states)

        # Operator diagnostics on the full-cell graph (noncommutativity witness).
        A = build_knn_graph(lin.X, k=cfg.k).adjacency.astype(float)
        P_dir = directed_propagator(A, lin.pseudotime, cfg.beta, cfg.tau)
        P_undir = directed_propagator(A, lin.pseudotime, 0.0, cfg.tau)
        op_nn_dir.append(nonnormality(P_dir))
        op_nn_undir.append(nonnormality(P_undir))
        basis = graph_fourier_basis(A, max(cfg.rank, 80))
        op_commutator.append(
            truncation_commutator_norm(basis, lin.X[:, 0], lin.X[:, 1]))
        # PD witness of the noncommutative truncated operator kernel, and the
        # verified spectral radius rho(alpha B) that certifies the closed-form solve.
        K = truncated_operator_kernel(basis, lin.X)
        op_kernel_mineig.append(kernel_min_eigenvalue(K))
        Bop = truncated_directed_operator(basis, P_dir, cfg.eps)
        op_spec_radius.append(spectral_radius(cfg.alpha * Bop))

        # Persistence barcode of the cell cloud.
        bc = h0_barcode(lin.X, k=cfg.k)
        for key, val in persistence_features(bc).items():
            pers[key].append(float(val))
        if s == 0:
            betti_curve_seed0 = bc.betti0_curve(betti_scales)
            spectrum_seed0 = [float(v) for v in basis.vals[:80]]

        # Forecasting under each split.
        splits = all_splits(lin, cfg.splits, label_fraction=cfg.label_fraction, seed=seed)
        if s == 0 and sweep_ranks:
            trunc_sweep_seed0 = _truncation_sweep(
                lin, cfg, splits[cfg.splits[0]], sweep_ranks)
        for kind, split in splits.items():
            if has_leakage(split):
                splits_clean = False
            res = _eval_forecasters(lin, split, cfg)
            for f in FORECASTERS:
                acc[kind][f].append(res[f]["accuracy"])
                ece[kind][f].append(res[f]["ece"])
                rare[kind][f].append(res[f]["rare_recall"])

        # Conformal calibration (exchangeable hold-out) and active sampling.
        conf_res = _conformal_eval(lin, cfg, np.random.default_rng(seed + 999))
        for key in conf_keys:
            conf[key].append(conf_res[key])
        lift = _active_sampling_lift(lin, cfg, np.random.default_rng(seed + 12345))
        for strat in active:
            active[strat]["rare_recall_after"].append(lift[strat]["rare_recall_after"])
            active[strat]["rare_query_rate"].append(lift[strat]["rare_query_rate"])

    # ---- aggregate -------------------------------------------------------- #
    by_split = {kind: {f: {
        "accuracy": metrics.summarize(acc[kind][f]),
        "ece": metrics.summarize(ece[kind][f]),
        "rare_recall": metrics.summarize(rare[kind][f]),
    } for f in FORECASTERS} for kind in cfg.splits}

    active_summary = {strat: {
        "rare_recall": metrics.summarize(active[strat]["rare_recall_after"]),
        "rare_query_rate": metrics.summarize(active[strat]["rare_query_rate"]),
    } for strat in active}

    def mean_over_splits(metric, forecaster):
        return float(np.mean([by_split[k][forecaster][metric]["mean"] for k in cfg.splits]))

    operator = {
        "nonnormality_directed": metrics.summarize(op_nn_dir),
        "nonnormality_undirected": metrics.summarize(op_nn_undir),
        "truncation_commutator_norm": metrics.summarize(op_commutator),
        "kernel_min_eigenvalue": metrics.summarize(op_kernel_mineig),
        "diffusion_spectral_radius": metrics.summarize(op_spec_radius),
        "spectrum": spectrum_seed0,
        "truncation_sweep": trunc_sweep_seed0,
    }
    conformal = {"target_coverage": round(1.0 - cfg.conformal_delta, 4),
                 "std_inductive": {key: metrics.summarize(conf[key]) for key in conf_keys}}
    persistence = {"scales": betti_scales, "betti0_curve": betti_curve_seed0,
                   **{key: metrics.summarize(pers[key]) for key in pf_keys}}

    summary: Dict = {
        "config": cfg.__dict__,
        "provenance": prov.finalize().to_dict(),
        "methods": FORECASTERS,
        "integrity": {
            "splits_clean": bool(splits_clean),
            # The directed propagator is non-normal, and directional reweighting
            # strictly increases non-normality over the undirected random walk
            # (which is itself non-normal through degree-asymmetric normalization).
            "propagator_nonnormal": bool(
                np.mean(op_nn_dir) > 1e-3 and np.mean(op_nn_dir) > np.mean(op_nn_undir)),
            "operator_kernel_psd": bool(np.min(op_kernel_mineig) > -1e-8),
            "diffusion_well_posed": bool(np.max(op_spec_radius) < 1.0),
            "rare_state_present": bool(0.0 < np.mean(rare_fracs) < 0.15),
        },
        "by_split": by_split,
        "operator": operator,
        "conformal": conformal,
        "persistence": persistence,
        "active_sampling": active_summary,
        "headline": {},
    }

    summary["headline"] = {
        # Clean inductive comparison (no transductive confound) -- the headline.
        "std_inductive_accuracy": round(mean_over_splits("accuracy", "std_inductive"), 6),
        "baseline_accuracy": round(mean_over_splits("accuracy", "baseline_knn"), 6),
        "inductive_accuracy_gain": round(
            mean_over_splits("accuracy", "std_inductive")
            - mean_over_splits("accuracy", "baseline_knn"), 6),
        "std_inductive_rare_recall": round(mean_over_splits("rare_recall", "std_inductive"), 6),
        "baseline_rare_recall": round(mean_over_splits("rare_recall", "baseline_knn"), 6),
        "rare_recall_gain_inductive": round(
            mean_over_splits("rare_recall", "std_inductive")
            - mean_over_splits("rare_recall", "baseline_knn"), 6),
        # Transductive operator vs prior label-propagation baseline.
        "std_transductive_accuracy": round(mean_over_splits("accuracy", "std_transductive"), 6),
        "label_prop_accuracy": round(mean_over_splits("accuracy", "label_prop"), 6),
        "std_transductive_rare_recall": round(mean_over_splits("rare_recall", "std_transductive"), 6),
        "label_prop_rare_recall": round(mean_over_splits("rare_recall", "label_prop"), 6),
        # Operator diagnostics, conformal, active sampling.
        "nonnormality_directed": operator["nonnormality_directed"]["mean"],
        "conformal_target": conformal["target_coverage"],
        "conformal_coverage": conformal["std_inductive"]["coverage"]["mean"],
        "ece_raw": conformal["std_inductive"]["ece_raw"]["mean"],
        "ece_calibrated": conformal["std_inductive"]["ece_calibrated"]["mean"],
        "active_persistence_rare_recall": active_summary["persistence"]["rare_recall"]["mean"],
        "random_rare_recall": active_summary["random"]["rare_recall"]["mean"],
        "active_rare_recall_gain": round(
            active_summary["persistence"]["rare_recall"]["mean"]
            - active_summary["random"]["rare_recall"]["mean"], 6),
        "n_cells": int(cfg.n_cells),
        "n_states": int(np.max(n_states_seen)) if n_states_seen else 0,
        "rare_state_fraction": round(float(np.mean(rare_fracs)), 6),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
