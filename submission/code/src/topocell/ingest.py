"""Real-data integration hooks -- honest stubs, NOT fabricated data.

The experiments in this repository run on the *synthetic* lineage from
``synthetic.py``, which is fully controlled and reproducible. This module is the
seam where a user plugs in **their own real single-cell data**. The functions
below intentionally raise ``NotImplementedError`` with precise instructions:
the package never invents biological measurements it did not load.

Plugging in real data means returning a ``synthetic.Lineage`` (or a duck-typed
object exposing the same fields: ``X``, ``branch``/``state``, ``pseudotime``,
``donor``, ``batch``, ``perturbation``, ``rare_state``) so the rest of the
pipeline -- graph build, splits, forecasting, active sampling -- works unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_anndata(
    path: str | Path,
    embedding_key: str = "X_pca",
    state_key: str = "cell_type",
    pseudotime_key: str = "dpt_pseudotime",
    donor_key: str = "donor_id",
    batch_key: str = "batch",
    perturbation_key: str = "perturbation",
):
    """Load a real ``.h5ad`` (AnnData) file as a ``Lineage``.

    This is a STUB. To enable it, install ``scanpy``/``anndata`` and implement
    the body, mapping AnnData fields to the ``Lineage`` schema::

        import anndata as ad
        from .synthetic import Lineage
        adata = ad.read_h5ad(path)
        X = adata.obsm[embedding_key]                      # (n_cells, n_dims)
        state = encode(adata.obs[state_key])               # int labels
        pseudotime = adata.obs[pseudotime_key].to_numpy()
        donor = encode(adata.obs[donor_key])
        batch = encode(adata.obs[batch_key])
        perturbation = encode(adata.obs[perturbation_key])
        rare_state = rarest_label(state)
        return Lineage(X=X, branch=state, state=state, pseudotime=pseudotime,
                       donor=donor, batch=batch, perturbation=perturbation,
                       rare_state=rare_state)

    Parameters
    ----------
    path : path to a ``.h5ad`` file.
    *_key : ``obs`` / ``obsm`` keys that name the embedding and metadata columns.

    Raises
    ------
    NotImplementedError
        Always, until you wire up your AnnData source. The reference experiments
        do not depend on this; they use the synthetic benchmark.
    """
    raise NotImplementedError(
        "load_anndata is an integration stub. Install scanpy/anndata, then map "
        "your AnnData embedding (obsm['X_pca']) and obs columns to the Lineage "
        "schema as shown in the docstring. The reference benchmark uses "
        "synthetic.make_lineage(); no real data is shipped or fabricated."
    )


def load_perturbation_screen(
    path: str | Path,
    control_label: str = "control",
    n_top_genes: Optional[int] = 2000,
):
    """Load a perturbation screen (e.g. Perturb-seq / CRISPR screen) as a Lineage.

    This is a STUB. A real implementation would read the screen, select highly
    variable genes, compute an embedding, and encode the perturbation target as
    the ``perturbation`` field so that the held-out-condition split forecasts an
    *unseen* perturbation. See ``load_anndata`` for the field mapping.

    Raises
    ------
    NotImplementedError
        Always, until wired up. No fabricated perturbation effects are returned.
    """
    raise NotImplementedError(
        "load_perturbation_screen is an integration stub. Wire it to your screen "
        "(scanpy + a perturbation-aware embedding) and encode the perturbed "
        "target into the `perturbation` field; the held-out-condition split then "
        "tests forecasting of an unseen perturbation."
    )
