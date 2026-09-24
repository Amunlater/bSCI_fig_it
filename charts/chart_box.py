"""Box plots.

Grouped box plots for ESTIMATE tumour-microenvironment scores, COMP expression
in the validation cohorts, and CD8+ cell density in COMP-low vs COMP-high
tumours.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "box"
TITLE = "Box plots (TME scores, COMP expression, CD8 density)"
REQUIRED = ("estimate_scores",)


def build(data):
    panels = []
    tables = {}

    if data.estimate_scores:
        def draw_est(ax):
            names, box_data, positions, colors = [], [], [], []
            for i, sn in enumerate(data.estimate_scores):
                names.append(sn)
                for j, grp in enumerate(("low", "high")):
                    box_data.append(data.estimate_scores[sn][grp])
                    positions.append(i * 3 - 0.4 + j * 0.8)
                    colors.append(C["low"] if grp == "low" else C["high"])
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                            widths=0.6, manage_ticks=False)
            for patch, cc in zip(bp["boxes"], colors):
                patch.set_facecolor(cc)
                patch.set_alpha(0.75)
            ax.set_xticks([i * 3 for i in range(len(names))])
            ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Score")
        tables["ESTIMATE"] = pd.DataFrame({
            sn: {grp: np.mean(data.estimate_scores[sn][grp]) for grp in ("low", "high")}
            for sn in data.estimate_scores}).T.reset_index(names="score")
        panels.append(("ESTIMATE scores", draw_est))

    if data.comp_expr_cohorts:
        for i, (cohort, grps) in enumerate(list(data.comp_expr_cohorts.items())[:3]):
            def draw(ax, cohort=cohort, grps=grps):
                labels = list(grps)
                vals = [grps[k] for k in labels]
                bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
                for j, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(C["grey"] if j == 0 else C["comp"])
                    patch.set_alpha(0.75)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylabel("COMP expression")
                ymax = max(np.max(v) for v in vals)
                ax.plot([1, 2], [ymax * 1.05] * 2, color="black", lw=0.8)
                ax.text(1.5, ymax * 1.1, "***", ha="center", fontsize=12)
            tables[f"COMP_{cohort}"] = pd.DataFrame(
                [{"cohort": cohort, "group": k, "mean": float(np.mean(grps[k]))} for k in grps])
            panels.append((f"COMP - {cohort}", draw))

    if data.cd8_density:
        def draw_cd8(ax):
            vals = [data.cd8_density["COMP_low"], data.cd8_density["COMP_high"]]
            bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor(C["low"])
            bp["boxes"][1].set_facecolor(C["high"])
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_alpha(0.7)
            for k, v in enumerate(vals):
                ax.scatter(np.ones(len(v)) * (k + 0.85) + np.random.normal(0, 0.05, len(v)),
                           v, color=[C["low"], C["high"]][k], alpha=0.85, zorder=5)
            ax.set_xticklabels(["COMP-low", "COMP-high"], fontsize=9)
            ax.set_ylabel("CD8+ cells / HPF")
            ymax = max(np.max(v) for v in vals)
            ax.plot([1, 2], [ymax * 1.05] * 2, color="black", lw=0.8)
        tables["CD8_density"] = pd.DataFrame({
            "group": ["COMP_low", "COMP_high"],
            "mean": [float(np.mean(data.cd8_density["COMP_low"])),
                     float(np.mean(data.cd8_density["COMP_high"]))]})
        panels.append(("CD8 density", draw_cd8))

    n = len(panels)
    fig, axes = subplot_grid(n, ncols=3, panel_size=(6, 5))
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()

    report = ("Box plots comparing ESTIMATE stromal/immune/purity scores by risk group, "
              "COMP expression between normal and tumour in each cohort, and CD8+ "
              "density by COMP status.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed=tables, report=report)
