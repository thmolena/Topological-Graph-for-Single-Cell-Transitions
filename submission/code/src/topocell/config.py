"""YAML experiment configuration.

The single knob set that defines a run. Each field maps to a parameter named in
the paper's Methods (``Synthetic lineage``): ``n_cells``/``n_features`` size the
cell-by-feature matrix; ``n_branches`` and ``noise`` shape the branching lineage
(synthetic.py); ``rare_fraction`` sizes the rare state; ``k`` is the
neighbourhood size of the kNN cell graph (graph.py); ``label_fraction`` is the
few-label annotation budget and ``active_budget`` the active-sampling query
budget (policy.py); ``splits`` selects the leakage-checked transfer protocols
(splits.py); and ``n_seeds`` controls the seed aggregation behind every reported
confidence interval. ``configs/full.yaml`` is the reported-scale configuration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 0
    n_cells: int = 400
    n_features: int = 8
    n_branches: int = 3
    n_donors: int = 4
    n_batches: int = 2
    n_perturbations: int = 3
    noise: float = 0.45
    rare_fraction: float = 0.05
    k: int = 15
    label_fraction: float = 0.1
    splits: List[str] = field(default_factory=lambda: ["donor", "batch", "time", "perturbation"])
    active_budget: int = 20
    n_seeds: int = 1
    # Spectral-truncated directed-diffusion operator (operators.py).
    rank: int = 80            # spectral-truncation level (band-limit)
    alpha: float = 0.9        # diffusion / clamping trade-off
    beta: float = 1.0         # pseudotime-direction strength (0 = undirected)
    eps: float = 0.6          # directed-walk laziness in (0, 1]
    tau: float = 0.2          # pseudotime scale of the directional kernel
    # Split-conformal calibration (conformal.py).
    conformal_delta: float = 0.1   # target miscoverage (coverage >= 1 - delta)
    cal_fraction: float = 0.4      # fraction of annotated cells held for calibration

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
