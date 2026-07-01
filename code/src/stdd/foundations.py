"""From-scratch foundations for graph-aware single-cell transition forecasting."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Single-cell matrix",
        "A dataset is a finite matrix of cells by features plus metadata such "
        "as donor, batch, perturbation, pseudotime and state.  synthetic.py "
        "creates a seeded branching lineage with a rare late state so every "
        "claim is reproducible without fabricating real biological data.",
    ),
    (
        "kNN graph",
        "graph.py builds a symmetric k-nearest-neighbour graph from feature "
        "space.  The graph is the substrate for label diffusion, density "
        "estimation and Betti-0 connectivity checks.",
    ),
    (
        "Topology",
        "Betti-0 is the number of connected components.  As k increases edges "
        "are added, so component count cannot increase.  The tests use this "
        "monotonicity as a topology sanity check; the paper does not overclaim "
        "higher-order persistent homology.",
    ),
    (
        "Label propagation",
        "policy.py solves a graph-smoothing problem: keep labels close to the "
        "few annotated cells while minimizing Dirichlet energy on graph edges.  "
        "The fixed-point iteration is a deterministic message-passing surrogate "
        "and is compared to a kNN classifier using the same labelled cells.",
    ),
    (
        "Leak-free splits",
        "splits.py holds out donors, batches, future pseudotime or entire "
        "perturbations.  The leakage checks make sure the model cannot memorize "
        "held-out group identities.",
    ),
    (
        "Active sampling",
        "The topology-aware querying rule combines prediction entropy with an "
        "inverse-density signal.  Rare states sit in sparse regions, so the "
        "policy is evaluated by rare-state recall gain over random querying.",
    ),
    (
        "Metrics",
        "metrics.py reports accuracy, expected calibration error and rare-state "
        "recall with seed-level confidence intervals.  The figures show lineage "
        "structure and forecast/active-sampling bars rather than decorative "
        "images.",
    ),
    (
        "Reproduction path",
        "Run scripts/reproduce_all.sh full or stdd-reproduce.  The runner "
        "writes summary.json, then scripts render table and vector PDF figure "
        "artifacts from that single source of truth.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
