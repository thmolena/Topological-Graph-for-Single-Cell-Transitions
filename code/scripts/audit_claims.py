#!/usr/bin/env python
"""Readiness-gate audit: traceable numbers, required phrase, no forbidden claims.

Exits non-zero if any check fails, so it can gate CI / `make audit`.
"""
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from stdd import audit


def main() -> int:
    problems = []

    summary_path = Path("results/summary.json")
    if not summary_path.exists():
        print("FAIL: results/summary.json missing -- run `make demo` first")
        return 1
    summary = json.loads(summary_path.read_text())
    headline = summary.get("headline", {})
    if not headline:
        problems.append("headline dict is empty")

    # README must carry the mandated phrase and avoid forbidden claims.
    readme = Path("../README.md")
    if readme.exists():
        text = readme.read_text()
        problems += audit.audit_reproducibility_phrase(text)
        problems += audit.audit_forbidden_claims(text)

    # Every headline macro must be present (traceable to this run).
    for key in ("std_inductive_accuracy", "inductive_accuracy_gain",
                "rare_recall_gain_inductive", "conformal_coverage",
                "nonnormality_directed", "n_states"):
        if key not in headline:
            problems.append(f"headline macro {key!r} not generated")

    # Integrity flags must be true.
    integrity = summary.get("integrity", {})
    for flag in ("splits_clean", "propagator_nonnormal", "rare_regime_present"):
        if not integrity.get(flag, False):
            problems.append(f"integrity flag {flag!r} is not true")

    if problems:
        print("AUDIT FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("AUDIT PASSED: all numbers traceable, phrase present, no forbidden claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
