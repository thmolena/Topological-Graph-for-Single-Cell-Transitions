"""Real-data integration hooks -- honest stubs, NOT fabricated data.

The experiments in this repository run on the *synthetic* phase sweep from
``synthetic.py``, which is fully controlled and reproducible. This module is the
seam where a user plugs in **their own real classical-shadow data** from a
programmable quantum simulator. The functions below intentionally raise
``NotImplementedError`` with precise instructions: the package never invents
measurements it did not load.

Plugging in real data means returning a ``synthetic.PhaseSweep`` (or a duck-typed
object exposing the same fields: ``X``, ``phase``, ``control``, ``device``,
``shot_batch``, ``hamiltonian``, ``rare_regime``) so the rest of the pipeline --
graph build, splits, classification, active sampling -- works unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_classical_shadows(
    path: str | Path,
    feature_key: str = "shadow_features",
    phase_key: str = "phase",
    control_key: str = "control",
    device_key: str = "device",
    shot_batch_key: str = "shot_batch",
    hamiltonian_key: str = "hamiltonian",
):
    """Load a real classical-shadow dataset as a ``PhaseSweep``.

    This is a STUB. To enable it, load your shadow-feature matrix and metadata
    (e.g. from an HDF5/npz produced by a shadow-tomography pipeline) and map them
    to the ``PhaseSweep`` schema: feature matrix ``X`` (n_states x n_features),
    the discrete ``phase`` label, the swept ``control`` parameter, and the
    ``device`` / ``shot_batch`` / ``hamiltonian`` condition ids; set
    ``rare_regime`` to the rarest phase label.

    Raises
    ------
    NotImplementedError
        Always, until you wire up your shadow-data source. The reference
        experiments use ``synthetic.make_phase_sweep()``; no real data is shipped
        or fabricated.
    """
    raise NotImplementedError(
        "load_classical_shadows is an integration stub. Load your shadow-feature "
        "matrix and metadata, then map them to the PhaseSweep schema as shown in "
        "the docstring. The reference benchmark uses synthetic.make_phase_sweep()."
    )


def load_perturbed_hamiltonian_scan(
    path: str | Path,
    control_label: str = "g",
    n_top_features: Optional[int] = 2000,
):
    """Load a perturbed-Hamiltonian control-parameter scan as a ``PhaseSweep``.

    This is a STUB. A real implementation would read the scan over the control
    parameter (e.g. a transverse field ``g``), select informative shadow
    observables, and encode the perturbed-Hamiltonian id into the ``hamiltonian``
    field so that the held-out-condition split classifies an *unseen* perturbation.

    Raises
    ------
    NotImplementedError
        Always, until wired up. No fabricated measurements are returned.
    """
    raise NotImplementedError(
        "load_perturbed_hamiltonian_scan is an integration stub. Wire it to your "
        "control-parameter scan and encode the perturbed-Hamiltonian id into the "
        "`hamiltonian` field; the held-out-condition split then tests an unseen "
        "perturbation."
    )
