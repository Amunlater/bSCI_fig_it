"""Kaplan-Meier survival curves.

Covers every survival panel: risk-group OS, COMP high/low OS across the
validation cohorts, and the TMB x risk stratification.  Unused grid cells are
removed so the figure hugs its content.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "survival"
TITLE = "Kaplan-Meier survival curves"
REQUIRED = ("survival_times", "survival_events", "risk_groups")


def _km(ax, times, events, label, color, ci=True):
    kmf = KaplanMeierFitter()
    kmf.fit(np.asarray(times, dtype=float), np.asarray(events).astype(bool), label=label)
    kmf.plot_survival_function(ax=ax, color=color, lw=2, ci_show=ci, ci_alpha=0.15)
    return kmf


def build(data):
    panels = []          # (title, draw(ax))
    record = []
    processed = {}

    # -- A: risk groups --------------------------------------------------
    def draw_risk(ax):
        t = np.asarray(data.survival_times, float)
        e = np.asarray(data.survival_events).astype(bool)
        g = np.asarray(data.risk_groups)
        for grp, col, lbl in (("Low", C["low"], "Low risk"), ("High", C["high"], "High risk")):
            m = g == grp
            if m.sum() == 0:
                continue
            kmf = _km(ax, t[m], e[m], lbl, col)
            record.append({"panel": "risk", "group": lbl, "n": int(m.sum()),
                           "events": int(e[m].sum()),
                           "median_survival": float(kmf.median_survival_time_)})
            processed[f"risk_{grp}"] = kmf.survival_function_
        ax.legend(fontsize=8, loc="lower left")
    panels.append(("A  Risk groups", draw_risk))

    # -- B-D: COMP cohorts ----------------------------------------------
    if data.comp_km:
        for idx, cohort in enumerate(list(data.comp_km)[:3]):
            def draw(ax, cohort=cohort):
                for grp, col, lbl in (("low", C["low"], "Low COMP"), ("high", C["high"], "High COMP")):
                    if grp not in data.comp_km[cohort]:
                        continue
                    t2, e2 = data.comp_km[cohort][grp]
                    kmf = _km(ax, t2, np.asarray(e2).astype(bool), lbl, col, ci=False)
                    record.append({"panel": cohort, "group": lbl, "n": int(len(t2)),
                                   "events": int(np.asarray(e2).sum()),
                                   "median_survival": float(kmf.median_survival_time_)})
                    processed[f"{cohort}_{grp}"] = kmf.survival_function_
                ax.legend(fontsize=7, loc="lower left")
            panels.append((f"{'BCD'[idx]}  {cohort}", draw))

    # -- E-F: TMB x risk -------------------------------------------------
    if data.tmb_km_data:
        tmb_colors = {"Low-risk/Low-TMB": C["low"], "Low-risk/High-TMB": C["comp"],
                      "High-risk/Low-TMB": C["high"], "High-risk/High-TMB": "#F39B7F"}

        def draw_all(ax):
            for lbl in data.tmb_km_data:
                t3, e3 = data.tmb_km_data[lbl]
                _km(ax, t3, np.asarray(e3).astype(bool), lbl, tmb_colors.get(lbl, C["grey"]), ci=False)
            ax.legend(fontsize=7, loc="lower left")

        def draw_sub(ax):
            for lbl in list(data.tmb_km_data)[:3]:
                t3, e3 = data.tmb_km_data[lbl]
                _km(ax, t3, np.asarray(e3).astype(bool), lbl, tmb_colors.get(lbl, C["grey"]), ci=False)
            ax.legend(fontsize=7, loc="lower left")

        panels.append(("E  TMB x risk", draw_all))
        panels.append(("F  TMB x risk vs clinical", draw_sub))

    fig, axes = subplot_grid(len(panels), ncols=3, panel_size=(6.5, 5))
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Overall Survival")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()

    km_table = pd.DataFrame(record)
    report = (
        "Kaplan-Meier overall-survival curves. Panel A stratifies by risk score; "
        "B-D by COMP expression in validation cohorts; E-F by combined TMB and "
        "risk status. Median survival per group is tabulated."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"km_summary": km_table},
                       processed=processed or {"note": "no curves"},
                       report=report)
