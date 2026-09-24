"""Violin plots.

Distribution of CAF senescence scores by COMP status and predicted drug IC50
between low- and high-risk patients.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "violin"
TITLE = "Violin plots (senescence, drug sensitivity)"
REQUIRED = ()
REQUIRED_ANY = ("drug_ic50", "senescence_scores")


def _violin(ax, values, labels, colors, ylabel, sig=None):
    parts = ax.violinplot(values, showmedians=True, showextrema=True)
    for body, cc in zip(parts["bodies"], colors):
        body.set_facecolor(cc)
        body.set_alpha(0.7)
    for key in ("cbars", "cmaxes", "cmins", "cmedians"):
        if key in parts:
            parts[key].set_color("black")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    if sig:
        ymax = max(np.max(v) for v in values)
        ax.plot([1, 2], [ymax * 1.05] * 2, color="black", lw=0.8)
        ax.text(1.5, ymax * 1.1, sig, ha="center", fontsize=10)


def build(data):
    panels = []
    tables = {}
    records = []

    if data.senescence_scores:
        def draw_sen(ax):
            vals = [data.senescence_scores["COMP_low"], data.senescence_scores["COMP_high"]]
            _violin(ax, vals, ["COMP-low", "COMP-high"], [C["low"], C["comp"]],
                    "Senescence score", sig="p < 2e-16")
        tables["senescence"] = pd.DataFrame({
            "group": ["COMP_low", "COMP_high"],
            "mean": [float(np.mean(data.senescence_scores["COMP_low"])),
                     float(np.mean(data.senescence_scores["COMP_high"]))]})
        for g in ("COMP_low", "COMP_high"):
            for v in data.senescence_scores[g]:
                records.append({"analysis": "senescence", "group": g, "value": v})
        panels.append(("Senescence by COMP", draw_sen))

    if data.drug_ic50:
        for drug, grps in data.drug_ic50.items():
            def draw(ax, drug=drug, grps=grps):
                low, high = np.asarray(grps["low"], float), np.asarray(grps["high"], float)
                _violin(ax, [low, high], ["Low risk", "High risk"], [C["low"], C["high"]],
                        "Predicted IC50", sig="n.s." if abs(low.mean() - high.mean()) < 0.2 else "***")
            for grp in ("low", "high"):
                for v in grps[grp]:
                    records.append({"analysis": "drug_IC50", "group": f"{drug}:{grp}", "value": v})
            tables[f"IC50_{drug}"] = pd.DataFrame({
                "group": ["low", "high"],
                "mean_IC50": [float(np.mean(grps["low"])), float(np.mean(grps["high"]))]})
            panels.append((f"{drug}", draw))

    n = len(panels)
    fig, axes = subplot_grid(n, ncols=3, panel_size=(6, 5))
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()

    all_values = pd.DataFrame(records)
    report = ("Violin plots of CAF senescence score by COMP status and of predicted "
              "drug IC50 between low- and high-risk patients across the tested agents.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed={"all_values": all_values}, report=report)
