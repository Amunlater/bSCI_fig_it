"""Forest plots - Cox proportional-hazards effect sizes (hazard ratios).

Consolidates every forest panel of the paper: signature-gene univariate Cox,
clinical univariate Cox and clinical multivariate Cox.  Reads ``gene_cox``
(``{gene: (HR, CI_low, CI_high, p)}``), ``clinical_cox_uv`` and
``clinical_cox_mv`` (``[(name, HR, lo, hi, p), ...]``).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "forest"
TITLE = "Forest plots (Cox regression)"
REQUIRED = ("gene_cox",)


def _forest(ax, rows, color, title):
    for i, (_, hr, lo, hi, _p) in enumerate(rows):
        ax.errorbar(hr, i, xerr=[[hr - lo], [hi - hr]], fmt="s",
                    color=color, capsize=3, markersize=7, lw=1.5)
        ax.text(hr + 0.08, i, f"{hr:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.axvline(1, color="gray", ls="--", alpha=0.6)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Hazard Ratio (95% CI)")
    ax.set_title(title, loc="left", fontweight="bold")


def build(data):
    group_defs = []
    if data.gene_cox:
        group_defs.append(([(g, *v) for g, v in data.gene_cox.items()],
                           C["high"], "A  Signature genes", "signature"))
    if data.clinical_cox_uv:
        group_defs.append((data.clinical_cox_uv, C["sig"],
                           "B  Clinical (univariate)", "clinical_uv"))
    if data.clinical_cox_mv:
        group_defs.append((data.clinical_cox_mv, C["comp"],
                           "C  Clinical (multivariate)", "clinical_mv"))
    if not group_defs:
        group_defs = [([("n/a", 1.0, 1.0, 1.0, 1.0)], C["grey"], "A  (no data)", "none")]

    records = []
    for rows, _, _, tag in group_defs:
        for nm, hr, lo, hi, pv in rows:
            records.append({"model": tag, "variable": nm, "HR": hr,
                            "CI_low": lo, "CI_high": hi, "p_value": pv})
    table = pd.DataFrame(records)

    n = len(group_defs)
    fig, axes = subplot_grid(n, ncols=n, panel_size=(6.5, 5))
    for ax, (rows, color, title, _) in zip(axes, group_defs):
        _forest(ax, rows, color, title)
    fig.tight_layout()

    report = (
        "Hazard ratios with 95% confidence intervals from Cox models. "
        "Panels: signature genes, clinical univariate, clinical multivariate.\n\n"
        f"Median signature-gene HR = {np.median([r['HR'] for r in records]):.2f}."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"hazard_ratios": table},
                       processed={"hazard_ratios": table},
                       report=report)
