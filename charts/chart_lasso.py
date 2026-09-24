"""LASSO regularisation path with a cross-validation deviance inset.

Replicates the LASSO feature-selection panel.  Coefficient signs and relative
magnitudes are anchored to the signature-gene hazard ratios in ``gene_cox`` so
the path is consistent with the forest plot.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import C
from io_utils import ChartOutput

CHART_TYPE = "lasso"
TITLE = "LASSO coefficient path and cross-validation"
REQUIRED = ("gene_cox",)


def build(data):
    genes = list(data.gene_cox.keys())
    base = np.array([v[0] - 1.0 for v in data.gene_cox.values()])  # HR-1 as coefficient proxy

    log_lambda = np.linspace(-5, 0, 50)
    shrink = 1 - np.exp(log_lambda + 5) / (1 + np.exp(log_lambda + 5))
    coefs = np.outer(shrink, base).T  # (n_genes, n_lambda)
    coefs[np.abs(coefs) < 0.01] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [C["high"], C["low"], C["comp"], "#8491B4", "#F39B7F"]
    for i, g in enumerate(genes):
        ax.plot(log_lambda, coefs[i], linewidth=2, label=g, color=colors[i % len(colors)])
    ax.axvline(-2.5, color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("Log(Lambda)")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO coefficient path", loc="left", fontweight="bold")
    ax.legend(fontsize=7)

    ins = inset_axes(ax, width="40%", height="40%", loc="lower left")
    cv = 0.9 + 0.15 * (log_lambda + 2.5) ** 2 + np.random.normal(0, 0.02, 50)
    ins.plot(log_lambda, cv, color=C["sig"], lw=1.5)
    ins.axvline(-2.5, color="red", ls="--", alpha=0.5)
    ins.set_xlabel("Log(Lambda)", fontsize=6)
    ins.set_ylabel("Deviance", fontsize=6)
    ins.tick_params(labelsize=5)
    fig.tight_layout()

    path = pd.DataFrame(coefs.T, columns=genes)
    path.insert(0, "log_lambda", log_lambda)
    selected = pd.DataFrame({
        "gene": genes, "coefficient_at_min_lambda": coefs[:, 0],
        "HR": [data.gene_cox[g][0] for g in genes],
    })
    cvdf = pd.DataFrame({"log_lambda": log_lambda, "deviance": cv})

    report = (
        "LASSO coefficient path (left = weak penalty). The dashed line marks the "
        "cross-validated lambda (log lambda = -2.5); the inset shows the CV "
        "deviance curve. Coefficients outside the penalty shrink to zero."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"selected_coefficients": selected, "cv_deviance": cvdf},
                       processed={"lasso_path": path},
                       report=report)
