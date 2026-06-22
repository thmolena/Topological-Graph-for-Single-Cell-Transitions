#!/usr/bin/env python
"""Generate all figures from results/summary.json."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from topocell import plotting


def main() -> None:
    summary = json.loads(Path("results/summary.json").read_text())
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plotting.fig_lineage(summary, out / "fig_lineage.pdf")
    plotting.fig_forecast_bars(summary, out / "fig_forecast.pdf")
    print(f"wrote {out}/fig_lineage.pdf, {out}/fig_forecast.pdf")


if __name__ == "__main__":
    main()
