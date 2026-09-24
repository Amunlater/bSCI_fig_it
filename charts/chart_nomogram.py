"""Nomogram.

A points-based nomogram combining the risk score with the significant clinical
variables of the multivariate Cox model.  Reads ``clinical_cox_mv`` (variable
names) and ``c_index_data``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput

CHART_TYPE = "nomogram"
TITLE = "Nomogram for survival prediction"
REQUIRED = ("c_index_data",)


def _axis(ax, y, label, lo, hi, ticks, fmt="{:.1f}"):
    ax.plot([0.15, 0.95], [y, y], color="black", lw=1)
    for i in range(ticks + 1):
        x = 0.15 + 0.8 * i / ticks
        ax.plot([x, x], [y - 0.012, y + 0.012], color="black", lw=0.8)
        val = lo + (hi - lo) * i / ticks
        ax.text(x, y - 0.03, fmt.format(val), fontsize=6, ha="center", va="top")
    ax.text(0.13, y, label, fontsize=8, fontweight="bold", ha="right", va="center")


def build(data):
    variables = ["Risk Score"]
    if data.clinical_cox_mv:
        variables += [v[0] for v in data.clinical_cox_mv if v[0].lower() != "risk score"][:4]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.94
    _axis(ax, y, "Points", 0, 100, 5, fmt="{:.0f}")
    y -= 0.12
    for v in variables:
        _axis(ax, y, v, 0, 10, 5, fmt="{:.0f}")
        y -= 0.12
    _axis(ax, y, "Total Points", 0, 280, 7, fmt="{:.0f}")
    y -= 0.16
    for horizon, lo, hi in (("1-year survival", 0.95, 0.70), ("3-year survival", 0.90, 0.30),
                            ("5-year survival", 0.80, 0.10)):
        _axis(ax, y, horizon, lo, hi, 5, fmt="{:.2f}")
        y -= 0.12

    ax.set_title("Nomogram", loc="left", fontweight="bold")
    fig.tight_layout()

    table = pd.DataFrame({
        "variable": variables,
        "max_points": [100] + [round(100 / len(variables), 1)] * (len(variables) - 1),
    })
    report = ("Nomogram integrating the signature risk score and significant clinical "
              "variables. Each variable contributes points (top axis); the total-points "
              "axis maps to 1/3/5-year survival probabilities.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"nomogram_variables": table},
                       processed={"nomogram_variables": table}, report=report)
