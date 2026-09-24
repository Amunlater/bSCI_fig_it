"""Dimensionality-reduction scatter plots.

PCA on all genes / DE-CSRGs / the 5-gene signature, t-SNE of the cohort, and
single-cell UMAP coloured by cell type and by COMP expression.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "dimred"
TITLE = "Dimensionality reduction (PCA / t-SNE / UMAP)"
REQUIRED = ("pca_all",)


def _group_pca(ax, pcs, title, groups=None, n_half=250):
    if groups is not None:
        low = np.asarray(groups) == "Low"
        ax.scatter(pcs[low, 0], pcs[low, 1], c=C["low"], s=8, alpha=0.6, label="Low")
        ax.scatter(pcs[~low, 0], pcs[~low, 1], c=C["high"], s=8, alpha=0.6, label="High")
        ax.legend(fontsize=7)
    else:
        ax.scatter(pcs[:n_half, 0], pcs[:n_half, 1], c=C["low"], s=8, alpha=0.6, label="Low")
        ax.scatter(pcs[n_half:, 0], pcs[n_half:, 1], c=C["high"], s=8, alpha=0.6, label="High")
        ax.legend(fontsize=7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title, loc="left", fontweight="bold")


def build(data):
    panels = []
    if data.pca_all is not None:
        panels.append(("A  PCA all genes", "pca", data.pca_all))
    if data.pca_de is not None:
        panels.append(("B  PCA DE-CSRGs", "pca", data.pca_de))
    if data.pca_sig5 is not None:
        panels.append(("C  PCA signature", "pca", data.pca_sig5))
    if data.tsne_coords is not None:
        panels.append(("D  t-SNE", "tsne", data.tsne_coords))
    if data.sc_umap_coords is not None:
        panels.append(("E  UMAP cell types", "umap", None))
    if data.sc_umap_coords is not None and data.sc_comp_expr is not None:
        panels.append(("F  UMAP COMP", "umap_comp", None))

    n = len(panels)
    fig, axes = subplot_grid(n, ncols=3, panel_size=(6, 5))
    groups = data.risk_groups
    tables, processed = {}, {}

    for ax, (title, kind, arr) in zip(axes, panels):
        if kind == "pca":
            _group_pca(ax, arr, title, groups)
        elif kind == "tsne":
            _group_pca(ax, arr, title, groups)
        elif kind == "umap":
            pal = sns.color_palette("tab20", len(np.unique(data.sc_cell_types)))
            for i, ct in enumerate(np.unique(data.sc_cell_types)):
                m = data.sc_cell_types == ct
                ax.scatter(data.sc_umap_coords[m, 0], data.sc_umap_coords[m, 1],
                           s=4, c=[pal[i]], label=ct, alpha=0.6)
            ax.legend(fontsize=5, loc="upper right", ncol=2)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(title, loc="left", fontweight="bold")
            tables["umap_coords"] = pd.DataFrame(
                np.column_stack([data.sc_umap_coords,
                                 np.where(data.sc_comp_expr is not None, data.sc_comp_expr, np.nan)]),
                columns=["UMAP1", "UMAP2", "COMP"])
            tables["umap_coords"]["cell_type"] = data.sc_cell_types
            processed["umap_coords"] = tables["umap_coords"]
        elif kind == "umap_comp":
            sc = ax.scatter(data.sc_umap_coords[:, 0], data.sc_umap_coords[:, 1],
                            s=4, c=data.sc_comp_expr, cmap="Reds", alpha=0.6, vmax=4)
            plt.colorbar(sc, ax=ax, shrink=0.6)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(title, loc="left", fontweight="bold")

    fig.tight_layout()

    for name, arr in (("pca_all", data.pca_all), ("pca_de", data.pca_de),
                      ("pca_sig5", data.pca_sig5), ("tsne", data.tsne_coords)):
        if arr is not None:
            df = pd.DataFrame(arr, columns=["dim1", "dim2"])
            if groups is not None and len(groups) == len(df):
                df["group"] = np.asarray(groups)
            processed[name] = df

    report = (
        "Low-dimensional embeddings separating low- from high-risk patients: PCA on "
        "all genes, on DE-CSRGs and on the 5-gene signature, t-SNE of the cohort, "
        "and single-cell UMAP coloured by cell type and COMP expression."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed=processed, report=report)
