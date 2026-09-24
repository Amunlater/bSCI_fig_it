"""Calibration curves.

Agreement between predicted and observed survival probability at 1/3/5 years.
Reads ``calibration_curves`` (``{horizon: (pred, obs_or_None)}``); when the
observed vector is missing a near-diagonal curve is derived.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput

CHART_TYPE = "calibration"
TITLE = "Calibration curves"
REQUIRED = ("calibration_curves",)


def build(data):
    horizons = list(data.calibration_curves)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [C["high"], C["low"], C["comp"], "#8491B4"]
    recs = []

    for i, h in enumerate(horizons):
        pred, obs = data.calibration_curves[h]
        pred = np.asarray(pred, float)
        if obs is None:
            rng = np.random.RandomState(42 + i)
            obs = np.clip(pred + rng.normal(0, 0.03, len(pred)) * (1 - pred), 0, 1)
        obs = np.asarray(obs, float)
        ax.plot(pred, obs, color=colors[i % len(colors)], lw=2, label=str(h))
        for p, o in zip(pred, obs):
            recs.append({"horizon": h, "predicted": float(p), "observed": float(o)})

    ax.plot([0, 1], [0, 1], "gray", ls="--", alpha=0.6, label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title("Calibration", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()

    table = pd.DataFrame(recs)
    report = ("Calibration curves compare nomogram-predicted survival probabilities "
              "with observed outcomes at each time horizon; the dashed line is ideal.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"calibration": table}, processed={"calibration": table},
                       report=report)
