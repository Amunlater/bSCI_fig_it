"""Bar charts.

A single home for every bar-style panel in the paper: performance metrics,
pathway enrichment, immune deconvolution, deconvolution comparisons, COMP
prioritisation, single-cell marker fractions and spatial statistics.  Only the
panels whose data are present are drawn.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "bar"
TITLE = "Bar charts (metrics, enrichment, immune infiltration)"
REQUIRED = ()
REQUIRED_ANY = ("gsea_high", "gsea_low", "cibersort", "tide_scores", "cms_counts",
                "comp_priority", "sc_comp_markers", "fib_comp_enrichment",
                "gsea_comp_high", "morans_i", "neighborhood_enrich", "ips_scores",
                "auc_comparison", "c_index_data")


def build(data):
    panels = []      # (title, closure(ax))
    tables = {}

    def barh(title, series, color, xlabel):
        def draw(ax):
            names, vals = list(series.keys()), list(series.values())
            cols = [color] * len(vals)
            ax.barh(range(len(names)), vals, color=cols, alpha=0.85, ec="white")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.axvline(0, color="gray", alpha=0.5)
            ax.set_xlabel(xlabel)
        tables[title] = pd.DataFrame({"name": list(series.keys()), "value": list(series.values())})
        return title, draw

    def bar(title, series, color, xlabel, rotate=0):
        def draw(ax):
            ax.bar(list(series.keys()), list(series.values()), color=color, ec="white")
            ax.set_ylabel(xlabel)
            ax.tick_params(axis="x", rotation=rotate)
        tables[title] = pd.DataFrame({"name": list(series.keys()), "value": list(series.values())})
        return title, draw

    def grouped(title, dct, ylabel, a="low", b="high", la="Low", lb="High", rotate=0):
        def draw(ax):
            keys = list(dct)
            low = [np.mean(dct[k][a]) for k in keys]
            high = [np.mean(dct[k][b]) for k in keys]
            x = np.arange(len(keys))
            w = 0.35
            ax.bar(x - w / 2, low, w, label=la, color=C["low"], ec="white")
            ax.bar(x + w / 2, high, w, label=lb, color=C["high"], ec="white")
            ax.set_xticks(x)
            ax.set_xticklabels(keys, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
        rows = [{"name": k, la: np.mean(dct[k][a]), lb: np.mean(dct[k][b])} for k in dct]
        tables[title] = pd.DataFrame(rows)
        return title, draw

    if data.auc_comparison:
        panels.append(barh("AUC comparison", data.auc_comparison, C["high"], "AUC"))
    if data.c_index_data:
        panels.append(bar("C-index", data.c_index_data, C["sig"], "C-index"))
    if data.gsea_high:
        panels.append(barh("GSEA high-risk", data.gsea_high, C["high"], "NES"))
    if data.gsea_low:
        panels.append(barh("GSEA low-risk", data.gsea_low, C["low"], "NES"))
    if data.cibersort:
        panels.append(grouped("CIBERSORT", data.cibersort, "Abundance", rotate=30))
    if data.tide_scores:
        panels.append(grouped("TIDE", data.tide_scores, "Score"))
    if data.cms_counts:
        panels.append(grouped("CMS subtypes", data.cms_counts, "Count"))
    if data.comp_priority:
        def draw_priority(ax):
            genes = list(data.comp_priority)
            crit = list(data.comp_priority[genes[0]])
            arr = np.array([[data.comp_priority[g][c] for c in crit] for g in genes], float)
            arr = (arr - arr.min(0)) / (arr.max(0) - arr.min(0) + 1e-10)
            x = np.arange(len(crit))
            w = 0.15
            cols = [C["high"], C["low"], C["comp"], "#8491B4", "#F39B7F"]
            for i, g in enumerate(genes):
                ax.bar(x + i * w - 2 * w, arr[i], w, label=g, color=cols[i % 5], ec="white")
            ax.set_xticks(x)
            ax.set_xticklabels(crit, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("Normalised score")
            ax.legend(fontsize=7)
        tables["COMP priority"] = pd.DataFrame(data.comp_priority).T
        panels.append(("COMP priority", draw_priority))
    if data.sc_comp_markers:
        panels.append(bar("scRNA co-expression", data.sc_comp_markers,
                          C["high"], "Fraction of COMP+ cells", rotate=30))
    if data.fib_comp_enrichment:
        panels.append(bar("CAF COMP enrichment", data.fib_comp_enrichment, C["high"], "Enrichment"))
    if data.gsea_comp_high:
        series = {k: v[0] for k, v in data.gsea_comp_high.items()}
        panels.append(barh("GSEA COMP-high CAF", series, C["high"], "NES"))
    if data.morans_i:
        series = {k: v[0] for k, v in data.morans_i.items()}
        panels.append(bar("Moran's I", series, C["high"], "Moran's I", rotate=30))
    if data.neighborhood_enrich:
        def draw_nb(ax):
            names = list(data.neighborhood_enrich)
            ax.bar(names, list(data.neighborhood_enrich.values()), color=C["high"], ec="white")
            ax.axhline(1, color="gray", ls="--", alpha=0.5)
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylabel("Fold enrichment")
        tables["Neighborhood"] = pd.DataFrame({"name": list(data.neighborhood_enrich),
                                                "fold": list(data.neighborhood_enrich.values())})
        panels.append(("Neighborhood enrichment", draw_nb))
    if data.ips_scores:
        panels.append(grouped("IPS", data.ips_scores, "IPS", rotate=20))

    n = len(panels)
    fig, axes = subplot_grid(n, ncols=4, panel_size=(5.5, 4.5))
    for ax, (title, draw) in zip(axes, panels):
        draw(ax)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout()

    report = (
        "Bar panels: model performance (AUC, C-index), pathway enrichment (GSEA), "
        "immune deconvolution (CIBERSORT, TIDE, IPS), molecular subtypes (CMS), "
        "COMP prioritisation, single-cell CAF co-expression/enrichment and spatial "
        "statistics (Moran's I, neighbourhood enrichment)."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed={k: v for k, v in tables.items()},
                       report=report)
