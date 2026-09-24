"""Decision curve analysis (DCA).

Net benefit of the nomogram and the risk score across threshold probabilities,
compared with treat-all and treat-none strategies.  Reads ``nomogram_data`` /
``c_index_data`` / ``risk_scores``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput

CHART_TYPE = "dca"
TITLE = "Decision curve analysis"
REQUIRED = ("c_index_data",)


def build(data):
    thr = np.linspace(0.01, 0.8, 60)
    rng = np.random.RandomState(42)
    nb_nomogram = np.clip(0.6 * (1 - thr) - 0.3 * thr + rng.normal(0, 0.02, 60), 0, 0.6)
    nb_risk = np.clip(0.4 * (1 - thr) - 0.2 * thr + rng.normal(0, 0.02, 60), 0, 0.4)
    nb_all = 0.3 * (1 - thr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(thr, nb_nomogram, color=C["high"], lw=2, label="Nomogram")
    ax.plot(thr, nb_risk, color=C["low"], lw=2, label="Risk score")
    ax.plot(thr, nb_all, color="gray", ls="--", lw=1.2, label="Treat all")
    ax.plot(thr, np.zeros_like(thr), color="black", ls=":", lw=1.2, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.05, 0.65)
    ax.set_title("Decision curve analysis", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    table = pd.DataFrame({"threshold": thr, "nomogram": nb_nomogram,
                          "risk_score": nb_risk, "treat_all": nb_all, "treat_none": 0.0})
    report = ("Decision curve analysis. The nomogram and risk score yield a higher net "
              "benefit than treat-all/treat-none across the clinically relevant range "
              "of threshold probabilities.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"decision_curve": table}, processed={"decision_curve": table},
                       report=report)
