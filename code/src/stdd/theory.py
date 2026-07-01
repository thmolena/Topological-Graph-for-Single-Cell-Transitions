"""From-principles guide for graph-based single-cell transition forecasting.

Data model
----------
synthetic.py creates a seeded branching lineage. Each cell is a feature vector
with metadata for branch/state, donor, batch, perturbation and pseudotime. A
small rare state is injected near a branch tip so rare-state recall can be tested
instead of assumed.

k-nearest-neighbour graph
-------------------------
graph.py builds a symmetric kNN graph from the cell features. The graph encodes
local manifold connectivity. Betti-0, the number of connected components, is used
as a simple topology sanity check; as k grows, components can merge but cannot
split.

Label propagation
-----------------
policy.py implements graph-smoothed label propagation. The mathematical object
is the graph Laplacian L = D - A, whose quadratic form measures how sharply a
label function changes across graph edges. Label propagation minimizes a
smoothness term plus a penalty for disagreeing with the few annotated cells.

Baseline and machine learning
-----------------------------
The baseline is a plain kNN classifier using the same annotated cells and the
same neighbourhood scale. The graph-smoothed method is therefore tested for the
value of diffusion over the graph, not for access to more labels. Active sampling
combines uncertainty with inverse local density to query cells likely to improve
rare-state recovery.

Splits and metrics
------------------
splits.py holds out donors, batches, time ranges or perturbations to test
transfer rather than memorization. metrics.py reports accuracy, expected
calibration error and rare-state recall with confidence intervals across fixed
seeds.

Reproduction
------------
runner.py writes results/summary.json as the source for every table and figure.
From code/:

    export PYTHONPATH=src
    make test
    bash scripts/reproduce_all.sh full
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
