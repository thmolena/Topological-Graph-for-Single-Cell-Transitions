"""Installed reproduction command for the graph-aware single-cell artifact.

The ``topocell-reproduce`` console entry point regenerates ``results/summary.json``,
the LaTeX/Markdown tables, and the figure PDFs from a deterministic, seeded run.
The configuration defaults to the reported scale (``configs/full.yaml``); pass
``--config configs/smoke.yaml`` for the laptop-scale smoke run, or ``--skip-run``
to regenerate tables and figures from an existing ``results/summary.json``.

Execution is delegated to the source-tree scripts so that an editable install and
a wheel install reproduce the same artifacts. Seeding is fixed in the configs and
in :mod:`topocell.seed`; ``OMP_NUM_THREADS`` is pinned to keep numeric results
stable across machines.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _has_artifact_tree(path: Path) -> bool:
    return (path / "scripts" / "run.py").is_file() and (path / "configs").is_dir()


def _code_dir() -> Path:
    """Return the artifact root that holds ``scripts/``, ``configs/`` and ``results/``.

    The pipeline drives the source-tree scripts and configs, which are not part of
    the importable package. The root is located, in order, from: the package layout
    (an editable / source-tree install), the current working directory, and its
    parents. Running ``topocell-reproduce`` from a wheel install therefore requires
    the working directory to be the ``code/`` artifact tree (or a subdirectory of it).
    """
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if _has_artifact_tree(candidate):
            return candidate
    raise FileNotFoundError(
        "Unable to locate the topocell artifact tree (scripts/run.py and configs/). "
        "Run topocell-reproduce from the 'code/' directory of the repository."
    )


def _run(*args: str) -> None:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run([sys.executable, *args], cwd=_code_dir(), env=env, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/full.yaml",
        help="experiment configuration (default: configs/full.yaml, reported scale)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="regenerate tables and figures from an existing results/summary.json",
    )
    args = parser.parse_args(argv)

    if not args.skip_run:
        _run("scripts/run.py", "--config", args.config, "--out", "results")
    _run("scripts/make_tables.py")
    _run("scripts/make_figures.py")


if __name__ == "__main__":
    main()
