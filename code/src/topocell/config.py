"""YAML experiment configuration."""
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

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
