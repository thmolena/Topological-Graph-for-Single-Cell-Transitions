"""End-to-end experiment driver.

Produces ``results/summary.json`` -- the single source of truth for every table,
figure and macro. For each random seed and each leakage-checked transfer protocol
(device / shot_batch / schedule / hamiltonian) the runner evaluates four
classifiers on an identical small annotated set:

  * ``baseline_knn``      -- a plain (inductive) kNN point classifier;
  * ``label_prop``        -- transductive symmetric label propagation;
  * ``std_transductive``  -- spectral-truncated *directed* diffusion, transductive;
  * ``std_inductive``     -- the same operator made *inductive* by an
                            out-of-sample Nystrom extension (the headline method).

It then (i) calibrates the inductive classifier by split-conformal prediction and
temperature scaling on an exchangeable hold-out; (ii) records operator
diagnostics -- the relative non-normality of the directed propagator against the
undirected control, a truncated-multiplication commutator norm, the positive-
definite operator-kernel eigenvalue, and the verified diffusion spectral radius;
(iii) computes the degree-0 persistence barcode of the phase diagram; and
(iv) compares random, density and persistence active sampling for rare-regime
recovery. Integrity gates assert the splits are leakage-free, the directed
propagator is non-normal, the operator kernel is positive definite, the diffusion
is well posed, and the rare regime is present.
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
from .persistence import h0_barcode, persistence_features
from .policy import (
    KNNForecaster,
    LabelPropagationForecaster,
    active_sample,
    recall_after_query,
)
from .graph import build_knn_graph
from .seed import RunProvenance, set_seed
from .splits import all_splits, has_leakage
from .synthetic import make_phase_sweep

FORECASTERS = ["baseline_knn", "label_prop", "std_transductive", "std_inductive"]


def _forecast(name, ps, split, cfg):
    if name == "baseline_knn":
        return KNNForecaster(k=cfg.k).predict(ps.X, ps.phase, split.labeled, split.test)
    if name == "label_prop":
        return LabelPropagationForecaster(k=cfg.k).predict(
            ps.X, ps.phase, split.labeled, split.test)
    if name == "std_transductive":
        return SpectralTruncatedDiffusion(
            k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(ps.X, ps.phase, split.labeled, split.test, ps.control)
    if name == "std_inductive":
        return InductiveSpectralTruncatedDiffusion(
            k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(ps.X, ps.phase, split.train, split.test,
                                 split.labeled, ps.control)
    raise ValueError(name)


def _eval_forecasters(ps, split, cfg) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    y_test = ps.phase[split.test]
    for name in FORECASTERS:
        fc = _forecast(name, ps, split, cfg)
        out[name] = {
            "accuracy": metrics.accuracy(y_test, fc.pred),
            "ece": metrics.expected_calibration_error(y_test, fc.pred, fc.confidence),
            "rare_recall": metrics.rare_state_recall(y_test, fc.pred, ps.rare_regime),
        }
    return out


def _conformal_eval(ps, cfg, rng) -> Dict[str, float]:
    """Conformal + temperature calibration of the inductive classifier on an
    exchangeable random hold-out (so the coverage guarantee applies)."""
    n = ps.n_states
    perm = rng.permutation(n)
    n_test = n // 4
    n_cal = n // 4
    test_idx = perm[:n_test]
    cal_idx = perm[n_test:n_test + n_cal]
    train_idx = perm[n_test + n_cal:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    cal_mask = np.zeros(n, bool); cal_mask[cal_idx] = True
    test_mask = np.zeros(n, bool); test_mask[test_idx] = True

    from .splits import _annotate
    labeled = _annotate(train_mask, ps, cfg.label_fraction, int(rng.integers(1 << 30)))

    model = InductiveSpectralTruncatedDiffusion(
        k=cfg.k, rank=cfg.rank, alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps, tau=cfg.tau)
    fc_cal = model.predict(ps.X, ps.phase, train_mask, cal_mask, labeled, ps.control)
    fc_test = model.predict(ps.X, ps.phase, train_mask, test_mask, labeled, ps.control)
    classes = fc_test.classes
    col = {int(c): j for j, c in enumerate(classes)}

    y_cal = ps.phase[cal_mask]
    y_test = ps.phase[test_mask]
    keep_cal = np.array([int(c) in col for c in y_cal])
    y_cal_idx = np.array([col[int(c)] for c in y_cal[keep_cal]])
    proba_cal = fc_cal.proba[keep_cal]

    T = fit_temperature(proba_cal, y_cal_idx)
    thr = conformal_threshold(proba_cal, y_cal_idx, delta=cfg.conformal_delta)
    sets = prediction_sets(fc_test.proba, thr)
    keep_test = np.array([int(c) in col for c in y_test])
    y_test_idx = np.array([col[int(c)] for c in y_test[keep_test]])
    rare_col = col.get(int(ps.rare_regime))
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


def _truncation_sweep(ps, cfg, split, ranks) -> Dict[str, List[float]]:
    """Inductive accuracy / rare-recall as a function of the truncation rank."""
    accs, rrs = [], []
    y_test = ps.phase[split.test]
    for r in ranks:
        fc = InductiveSpectralTruncatedDiffusion(
            k=cfg.k, rank=int(r), alpha=cfg.alpha, beta=cfg.beta, eps=cfg.eps,
            tau=cfg.tau).predict(ps.X, ps.phase, split.train, split.test,
                                 split.labeled, ps.control)
        accs.append(metrics.accuracy(y_test, fc.pred))
        rrs.append(metrics.rare_state_recall(y_test, fc.pred, ps.rare_regime))
    return {"ranks": [int(r) for r in ranks], "accuracy": accs, "rare_recall": rrs}


def _active_sampling_lift(ps, cfg, rng) -> Dict[str, Dict[str, float]]:
    n = ps.n_states
    seed_mask = np.zeros(n, dtype=bool)
    seed_idx = rng.choice(n, size=max(10, n // 10), replace=False)
    seed_mask[seed_idx] = True
    results = {}
    for strategy in ("persistence", "topology", "random"):
        queried = active_sample(ps.X, ps.phase, seed_mask, cfg.active_budget, k=cfg.k,
                                strategy=strategy,
                                rng=np.random.default_rng(int(rng.integers(1 << 30))))
        _, after = recall_after_query(ps.X, ps.phase, seed_mask, queried,
                                      ps.rare_regime, k=cfg.k)
        rare_hit = float(np.mean(ps.phase[queried] == ps.rare_regime)) if queried.size else 0.0
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
    rare_fracs, n_phases_seen = [], []
    betti_scales = list(np.round(np.linspace(1.0, 3.5, 6), 4))
    betti_curve_seed0: List[int] = []
    spectrum_seed0: List[float] = []
    trunc_sweep_seed0: Dict = {}
    sweep_ranks = [r for r in (8, 16, 32, 48, 64, 80, 120, 160, 240) if r < cfg.n_states // 4]

    for s in range(cfg.n_seeds):
        seed = cfg.seed + s
        ps = make_phase_sweep(
            n_states=cfg.n_states, n_features=cfg.n_features,
            n_phase_branches=cfg.n_phase_branches, n_devices=cfg.n_devices,
            n_shot_batches=cfg.n_shot_batches, n_hamiltonians=cfg.n_hamiltonians,
            noise=cfg.noise, rare_fraction=cfg.rare_fraction, seed=seed)
        rare_fracs.append(ps.phase_frequencies().get(ps.rare_regime, 0.0))
        n_phases_seen.append(ps.n_phases)

        A = build_knn_graph(ps.X, k=cfg.k).adjacency.astype(float)
        P_dir = directed_propagator(A, ps.control, cfg.beta, cfg.tau)
        P_undir = directed_propagator(A, ps.control, 0.0, cfg.tau)
        op_nn_dir.append(nonnormality(P_dir))
        op_nn_undir.append(nonnormality(P_undir))
        basis = graph_fourier_basis(A, max(cfg.rank, 80))
        op_commutator.append(truncation_commutator_norm(basis, ps.X[:, 0], ps.X[:, 1]))
        K = truncated_operator_kernel(basis, ps.X)
        op_kernel_mineig.append(kernel_min_eigenvalue(K))
        Bop = truncated_directed_operator(basis, P_dir, cfg.eps)
        op_spec_radius.append(spectral_radius(cfg.alpha * Bop))

        bc = h0_barcode(ps.X, k=cfg.k)
        for key, val in persistence_features(bc).items():
            pers[key].append(float(val))
        if s == 0:
            betti_curve_seed0 = bc.betti0_curve(betti_scales)
            spectrum_seed0 = [float(v) for v in basis.vals[:80]]

        splits = all_splits(ps, cfg.splits, label_fraction=cfg.label_fraction, seed=seed)
        if s == 0 and sweep_ranks:
            trunc_sweep_seed0 = _truncation_sweep(ps, cfg, splits[cfg.splits[0]], sweep_ranks)
        for kind, split in splits.items():
            if has_leakage(split):
                splits_clean = False
            res = _eval_forecasters(ps, split, cfg)
            for f in FORECASTERS:
                acc[kind][f].append(res[f]["accuracy"])
                ece[kind][f].append(res[f]["ece"])
                rare[kind][f].append(res[f]["rare_recall"])

        conf_res = _conformal_eval(ps, cfg, np.random.default_rng(seed + 999))
        for key in conf_keys:
            conf[key].append(conf_res[key])
        lift = _active_sampling_lift(ps, cfg, np.random.default_rng(seed + 12345))
        for strat in active:
            active[strat]["rare_recall_after"].append(lift[strat]["rare_recall_after"])
            active[strat]["rare_query_rate"].append(lift[strat]["rare_query_rate"])

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
            "propagator_nonnormal": bool(
                np.mean(op_nn_dir) > 1e-3 and np.mean(op_nn_dir) > np.mean(op_nn_undir)),
            "operator_kernel_psd": bool(np.min(op_kernel_mineig) > -1e-8),
            "diffusion_well_posed": bool(np.max(op_spec_radius) < 1.0),
            "rare_regime_present": bool(0.0 < np.mean(rare_fracs) < 0.15),
        },
        "by_split": by_split,
        "operator": operator,
        "conformal": conformal,
        "persistence": persistence,
        "active_sampling": active_summary,
        "headline": {},
    }

    summary["headline"] = {
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
        "std_transductive_accuracy": round(mean_over_splits("accuracy", "std_transductive"), 6),
        "label_prop_accuracy": round(mean_over_splits("accuracy", "label_prop"), 6),
        "std_transductive_rare_recall": round(mean_over_splits("rare_recall", "std_transductive"), 6),
        "label_prop_rare_recall": round(mean_over_splits("rare_recall", "label_prop"), 6),
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
        "n_states": int(cfg.n_states),
        "n_phases": int(np.max(n_phases_seen)) if n_phases_seen else 0,
        "rare_regime_fraction": round(float(np.mean(rare_fracs)), 6),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
