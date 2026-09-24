"""Receiver operating characteristic (ROC) curves.

Time-dependent ROC for the risk model (1/3/5-year) plus COMP-expression ROC in
each validation cohort.  Reads ``roc_1yr``/``roc_3yr``/``roc_5yr`` and
``comp_roc`` (``{cohort: (fpr, tpr[, auc])}``).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "roc"
TITLE = "ROC curves"
REQUIRED = ("roc_1yr", "roc_3yr", "roc_5yr")


def _auc(fpr, tpr):
    return float(np.trapz(np.asarray(tpr, float), np.asarray(fpr, float)))


def _roc_panel(ax, fpr, tpr, auc, color, title):
    ax.plot(fpr, tpr, color=color, lw=2, label=f"AUC={auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    ax.plot([0, 1], [0, 1], "gray", ls="--", alpha=0.5)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")


def build(data):
    # normalise every curve to (title, fpr, tpr, label, color)
    curves = [
        ("A  1-year", *data.roc_1yr, "1-year", C["high"]),
        ("B  3-year", *data.roc_3yr, "3-year", C["low"]),
        ("C  5-year", *data.roc_5yr, "5-year", C["comp"]),
    ]
    if data.comp_roc:
        for i, (cohort, tup) in enumerate(list(data.comp_roc.items())[:3]):
            fpr, tpr = tup[0], tup[1]
            curves.append((f"{'DEF'[i]}  {cohort}", fpr, tpr, cohort, "#00A087"))

    n = len(curves)
    fig, axes = subplot_grid(n, ncols=3, panel_size=(5, 5))

    recs, processed = [], {}
    for ax, (title, fpr, tpr, label, color) in zip(axes, curves):
        fpr, tpr = np.asarray(fpr, float), np.asarray(tpr, float)
        auc = _auc(fpr, tpr)
        _roc_panel(ax, fpr, tpr, auc, color, title)
        recs.append({"curve": label, "auc": round(auc, 3)})
        processed[f"roc_{label}"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    fig.tight_layout()

    table = pd.DataFrame(recs)
    report = (
        "ROC curves with area under the curve. Panels A-C are time-dependent ROCs "
        "for the risk score at 1/3/5 years; D-F are COMP-expression ROCs in the "
        "validation cohorts."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"auc_summary": table}, processed=processed, report=report)
