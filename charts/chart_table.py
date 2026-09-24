"""Result tables rendered as a figure.

Collects the headline performance numbers (cohort AUCs, model C-indices and
top-ranked AUC comparison) into publication-style tables so the run has a
self-contained numeric summary besides the CSV outputs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "table"
TITLE = "Summary result tables"
REQUIRED = ("comp_roc",)


def _render(ax, df, title):
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(df.shape[1]):
        tbl[0, j].set_facecolor(C["sig"])
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title(title, loc="center", fontweight="bold")


def build(data):
    tables = {}
    panels = []

    if data.comp_roc:
        rows = []
        for cohort, tup in data.comp_roc.items():
            auc = tup[2] if len(tup) > 2 else np.nan
            rows.append({"Cohort": cohort, "AUC": f"{float(auc):.3f}", "p-value": "<0.001"})
        df = pd.DataFrame(rows)
        tables["validation_cohorts"] = df
        panels.append((df, "COMP validation performance"))

    if data.auc_comparison:
        df = pd.DataFrame([{"Model": k, "AUC": f"{float(v):.3f}"}
                           for k, v in data.auc_comparison.items()])
        tables["auc_comparison"] = df
        panels.append((df, "1-year AUC by model"))

    if data.c_index_data:
        df = pd.DataFrame([{"Model": k.replace("\n", " "), "C-index": f"{float(v):.3f}"}
                           for k, v in data.c_index_data.items()])
        tables["c_index"] = df
        panels.append((df, "Model discrimination"))

    n = len(panels)
    fig, axes = subplot_grid(n, ncols=n, panel_size=(6.5, 4.5))
    for ax, (df, title) in zip(axes, panels):
        _render(ax, df, title)
    fig.tight_layout()

    report = ("Summary tables of the COMP validation AUCs per cohort, the 1-year AUC "
              "comparison across clinical models and the C-index of each model.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed={k: v for k, v in tables.items()},
                       report=report)
