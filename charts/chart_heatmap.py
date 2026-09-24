"""Heatmaps.

Panel A renders the signature-gene expression matrix ordered by risk score
(high risk on the right).  Panel B renders the CellChat ligand-receptor
communication matrix.  Either panel is drawn alone if the other is unavailable.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "heatmap"
TITLE = "Gene / interaction heatmaps"
REQUIRED = ()
REQUIRED_ANY = ("gene_expr", "cellchat_matrix")


def build(data):
    has_expr = data.gene_expr is not None and data.risk_scores is not None
    has_chat = data.cellchat_matrix is not None

    kinds = (["expr"] if has_expr else []) + (["chat"] if has_chat else [])
    n = len(kinds)
    fig, axes = subplot_grid(n, ncols=max(n, 1), panel_size=(9, 7))

    tables, processed = {}, {}
    report = []

    idx = 0
    if has_expr:
        genes = list(data.gene_expr.columns)
        order = np.argsort(np.asarray(data.risk_scores))
        z = data.gene_expr.iloc[order][genes].values.astype(float)
        z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-9)
        z = np.clip(z, -2, 2)
        sns.heatmap(z.T, ax=axes[idx], cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                    yticklabels=genes, xticklabels=False,
                    cbar_kws={"label": "z-score", "shrink": 0.7})
        axes[idx].set_title("A  Signature gene expression (by risk)", loc="left", fontweight="bold")
        axes[idx].set_xlabel("Patients (sorted by risk score)")
        tables["expression_zscore"] = pd.DataFrame(z, columns=genes)
        processed["expression_zscore"] = tables["expression_zscore"]
        report.append("Panel A: z-scored signature-gene expression ordered by risk.")
        idx += 1

    if has_chat:
        ax = axes[idx]
        sns.heatmap(data.cellchat_matrix, ax=ax, xticklabels=data.cellchat_pathways,
                    yticklabels=data.cellchat_pairs, cmap="Reds",
                    cbar_kws={"shrink": 0.7}, linewidths=0.5)
        ax.set_xticklabels(data.cellchat_pathways, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{'B' if has_expr else 'A'}  CellChat ligand-receptor",
                     loc="left", fontweight="bold")
        chat_df = pd.DataFrame(data.cellchat_matrix, index=data.cellchat_pairs,
                               columns=data.cellchat_pathways)
        tables["cellchat_matrix"] = chat_df.reset_index(names="sender_to_receiver")
        processed["cellchat_matrix"] = chat_df
        report.append("CellChat ligand-receptor communication strength.")
        idx += 1

    fig.tight_layout()

    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed=processed,
                       report="\n\n".join(report) or "No heatmap data available.")
