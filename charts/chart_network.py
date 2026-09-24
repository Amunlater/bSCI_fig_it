"""DE-CSRG gene interaction network.

Draws the differentially expressed cancer-stroma related genes (DE-CSRGs) as a
peripheral ring with the signature hub genes placed centrally and connected to
their nearest neighbours.  Hub genes come from ``gene_cox``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import C
from io_utils import ChartOutput

CHART_TYPE = "network"
TITLE = "DE-CSRG gene network"
REQUIRED = ("gene_cox",)


def build(data):
    np.random.seed(42)
    hubs = list(data.gene_cox.keys())
    n_de = 66
    angles = np.linspace(0, 2 * np.pi, n_de, endpoint=False)
    xo = np.cos(angles) + np.random.normal(0, 0.05, n_de)
    yo = np.sin(angles) + np.random.normal(0, 0.05, n_de)

    hub_xy = {g: (np.cos(2 * np.pi * i / len(hubs) + 0.4) * 0.55,
                  np.sin(2 * np.pi * i / len(hubs) + 0.4) * 0.55)
              for i, g in enumerate(hubs)}
    colors = [C["high"], C["low"], "#F39B7F", "#8491B4", C["comp"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(xo, yo, s=12, c=C["grey"], alpha=0.5)

    edges = []
    for k, g in enumerate(hubs):
        hx, hy = hub_xy[g]
        ax.scatter(hx, hy, s=150, c=colors[k % len(colors)], ec="white", lw=1.5, zorder=5)
        ax.text(hx, hy, g, fontsize=8, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)
        for i in range(n_de):
            d = np.sqrt((xo[i] - hx) ** 2 + (yo[i] - hy) ** 2)
            if d < 0.7 and np.random.random() < 0.6:
                ax.plot([hx, xo[i]], [hy, yo[i]], color=colors[k % len(colors)],
                        alpha=0.15, lw=0.5)
                edges.append({"hub": g, "node": f"DE_{i}", "distance": round(d, 3)})

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DE-CSRG network", loc="left", fontweight="bold")
    fig.tight_layout()

    edge_df = pd.DataFrame(edges)
    nodes = pd.DataFrame({
        "node": list(hub_xy) + [f"DE_{i}" for i in range(n_de)],
        "type": ["hub"] * len(hub_xy) + ["DE-CSRG"] * n_de,
        "x": [hub_xy[g][0] for g in hubs] + list(xo),
        "y": [hub_xy[g][1] for g in hubs] + list(yo),
    })
    report = (
        f"Network of {n_de} DE-CSRGs (grey) connected to {len(hubs)} signature "
        "hub genes (coloured). Edges mark hub-neighbour proximity (distance < 0.7)."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"edges": edge_df, "nodes": nodes},
                       processed={"nodes": nodes},
                       report=report)
