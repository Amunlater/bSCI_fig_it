"""Unified output port.

Every chart returns a :class:`ChartOutput`; the :class:`OutputManager` is the
single place that writes results to disk.  One chart therefore always produces:

* ``figures/<type>.png``      - the rendered chart
* ``tables/<type>__*.csv``    - the numbers plotted in the chart
* ``processed/<type>__*``     - the processed data behind those numbers
* ``reports/<type>.md``       - a short methods/reading note

A run-level ``reports/summary_report.md`` and ``manifest.csv`` are written too.
"""
import os
import json
import math
import pickle
import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_SUBDIRS


def subplot_grid(n, ncols=3, panel_size=(5.5, 4.5)):
    """Create exactly ``n`` axes in a ``ncols``-column grid, no empty cells.

    Unused axes are *deleted* (``fig.delaxes``) rather than turned off, so
    ``tight_layout``/``bbox_inches`` reclaims the space and the saved figure has
    no large blank regions.  Panel dimensions scale with the grid size.
    """
    n = max(int(n), 1)
    ncols = max(1, min(int(ncols), n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             squeeze=False)
    axes = list(axes.ravel())
    for extra in axes[n:]:
        fig.delaxes(extra)
    axes = axes[:n]
    # remember the grid so the output layer can centre a partial last row
    fig._grid_axes = axes
    fig._grid_cap = ncols
    return fig, axes


def finalize_layout(fig):
    """Tight-layout a grid figure and horizontally centre a partial last row."""
    fig.tight_layout()
    axes = getattr(fig, "_grid_axes", None)
    cap = getattr(fig, "_grid_cap", 0)
    if not axes or not cap or cap <= 1 or len(axes) % cap == 0:
        return
    last = len(axes) % cap
    start = len(axes) - last
    if len(axes) >= cap + 1:
        dx = axes[1].get_position().x0 - axes[0].get_position().x0
    else:
        dx = axes[0].get_position().width
    shift = (cap - last) / 2.0 * dx
    for ax in axes[start:]:
        p = ax.get_position()
        ax.set_position([p.x0 + shift, p.y0, p.width, p.height])


@dataclass
class ChartOutput:
    chart_type: str
    title: str = ""
    figure: Optional[Any] = None
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    processed: Dict[str, Any] = field(default_factory=dict)
    report: str = ""


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


class OutputManager:
    def __init__(self, out_dir, fmt="png"):
        self.root = os.path.abspath(out_dir)
        self.fmt = fmt
        self.dirs = {k: os.path.join(self.root, k) for k in OUTPUT_SUBDIRS}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.manifest = []

    # -- writers ---------------------------------------------------------
    def _write_figure(self, out):
        if out.figure is None:
            return None
        finalize_layout(out.figure)
        path = os.path.join(self.dirs["figures"], f"{_safe(out.chart_type)}.{self.fmt}")
        out.figure.savefig(path, dpi=300, bbox_inches="tight")
        return path

    def _write_tables(self, out):
        paths = []
        for name, df in out.tables.items():
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            path = os.path.join(self.dirs["tables"], f"{_safe(out.chart_type)}__{_safe(name)}.csv")
            df.to_csv(path, index=False)
            paths.append(path)
        return paths

    def _write_processed(self, out):
        paths = []
        for name, obj in out.processed.items():
            base = os.path.join(self.dirs["processed"], f"{_safe(out.chart_type)}__{_safe(name)}")
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(base + ".csv", index=False)
                paths.append(base + ".csv")
            elif isinstance(obj, np.ndarray):
                np.save(base + ".npy", obj)
                paths.append(base + ".npy")
            elif isinstance(obj, (dict, list, tuple, str, int, float, bool)) or obj is None:
                try:
                    with open(base + ".json", "w", encoding="utf-8") as f:
                        json.dump(obj, f, indent=2, default=str)
                    paths.append(base + ".json")
                except (TypeError, ValueError):
                    with open(base + ".pkl", "wb") as f:
                        pickle.dump(obj, f)
                    paths.append(base + ".pkl")
            else:
                with open(base + ".pkl", "wb") as f:
                    pickle.dump(obj, f)
                paths.append(base + ".pkl")
        return paths

    def _write_report(self, out):
        if not (out.report or out.title):
            return None
        path = os.path.join(self.dirs["reports"], f"{_safe(out.chart_type)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {out.title or out.chart_type}\n\n")
            f.write(f"- chart type: `{out.chart_type}`\n")
            f.write(f"- generated: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
            f.write(out.report.strip() + "\n")
        return path

    # -- public ----------------------------------------------------------
    def write(self, out: ChartOutput):
        fig = self._write_figure(out)
        tables = self._write_tables(out)
        processed = self._write_processed(out)
        report = self._write_report(out)
        self.manifest.append({
            "chart_type": out.chart_type,
            "title": out.title,
            "figure": fig or "",
            "n_tables": len(tables),
            "n_processed": len(processed),
            "report": report or "",
        })
        return {"figure": fig, "tables": tables, "processed": processed, "report": report}

    def write_summary(self, outputs, input_info=None, data=None):
        rows = list(self.manifest)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(self.root, "manifest.csv"), index=False)

        lines = ["# figIndividual - analysis summary", "",
                 f"- generated: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
                 f"- output root: `{self.root}`",
                 f"- charts written: {len(outputs)}", ""]
        if input_info:
            lines += ["## Input", ""]
            for k, v in input_info.items():
                lines.append(f"- **{k}**: `{v}`")
            lines.append("")
        if data is not None:
            try:
                keys = data.filled()
            except AttributeError:  # duck-typed PaperData without .filled()
                keys = sorted(k for k, v in vars(data).items() if v is not None)
            lines += ["## Data fields available", "",
                      ", ".join(f"`{k}`" for k in keys), ""]
        lines += ["## Charts", ""]
        for out in outputs:
            lines.append(f"- `{out.chart_type}` - {out.title}")
        lines.append("")
        path = os.path.join(self.dirs["reports"], "summary_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
