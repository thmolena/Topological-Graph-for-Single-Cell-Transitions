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
    plotting.fig_schematic(summary, out / "fig_schematic.pdf")
    plotting.fig_lineage(summary, out / "fig_lineage.pdf")
    plotting.fig_forecast_bars(summary, out / "fig_forecast.pdf")
    plotting.fig_operator(summary, out / "fig_operator.pdf")
    plotting.fig_calibration(summary, out / "fig_calibration.pdf")
    print(f"wrote {out}/fig_schematic.pdf, {out}/fig_lineage.pdf, "
          f"{out}/fig_forecast.pdf, {out}/fig_operator.pdf, {out}/fig_calibration.pdf")


if __name__ == "__main__":
    main()
