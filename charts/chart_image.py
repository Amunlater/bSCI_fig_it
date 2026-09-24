"""Multiplex immunofluorescence and IHC images.

Renders representative marker fields (COMP, a-SMA, FAP, CD47, CD8) as synthetic
microscopy-like images.  These are illustrative reconstructions of the imaging
panels; the quantitative counterpart (CD8 density) is tabulated from
``cd8_density`` when available.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from config import C
from io_utils import ChartOutput

CHART_TYPE = "image"
TITLE = "Multiplex immunofluorescence / IHC"
REQUIRED = ("cd8_density",)


def _sim_ifield(size=100, n_objects=20, intensity_range=(0.3, 1.0), bg=0.05, seed=None):
    rng = np.random.RandomState(seed)
    img = np.ones((size, size)) * bg + rng.normal(0, 0.02, (size, size))
    yv, xv = np.arange(size), np.arange(size)
    for _ in range(n_objects):
        cx, cy = rng.randint(10, size - 10, 2)
        r = rng.randint(4, 12)
        obj = rng.uniform(*intensity_range)
        dist2 = (xv[np.newaxis, :] - cx) ** 2 + (yv[:, np.newaxis] - cy) ** 2
        mask = dist2 < r ** 2
        img[mask] = np.minimum(img[mask] + obj * np.exp(-dist2[mask] / (2 * (r / 2) ** 2)), 1.0)
    return np.clip(img, 0, 1)


def build(data):
    fig = plt.figure(figsize=(16, 18))
    np.random.seed(42)

    channels = [("A  COMP (green)", "Greens", (0.5, 1.0)),
                ("B  a-SMA (red)", "Reds", (0.4, 0.9)),
                ("C  FAP (cyan)", "Blues", (0.3, 0.8))]
    for i, (title, cmap, inten) in enumerate(channels):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.imshow(_sim_ifield(200, 25, inten, 0.02, seed=i), cmap=cmap)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(4, 3, (4, 6))
    comp = _sim_ifield(200, 22, (0.5, 1.0), 0.02, seed=10)
    cd47 = _sim_ifield(200, 20, (0.4, 0.9), 0.02, seed=11)
    rgb = np.zeros((200, 200, 3))
    rgb[:, :, 1] = comp
    rgb[:, :, 0] = cd47
    ax.imshow(rgb)
    ax.set_title("D  COMP(green) + CD47(red)", loc="left", fontweight="bold", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(4, 3, (7, 9))
    brown = LinearSegmentedColormap.from_list("brown", ["white", "#8B4513"])
    ax.imshow(_sim_ifield(200, 30, (0.3, 0.8), 0.05, seed=12), cmap=brown)
    ax.set_title("E  CD8 IHC", loc="left", fontweight="bold", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(4, 3, (10, 12))
    vals = [data.cd8_density["COMP_low"], data.cd8_density["COMP_high"]]
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C["low"])
    bp["boxes"][1].set_facecolor(C["high"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_alpha(0.7)
    for i, v in enumerate(vals):
        ax.scatter(np.ones(len(v)) * (i + 0.85) + np.random.normal(0, 0.05, len(v)),
                   v, color=[C["low"], C["high"]][i], alpha=0.85, zorder=5)
    ax.set_xticklabels(["COMP-low", "COMP-high"], fontsize=10)
    ax.set_ylabel("CD8+ cells / HPF")
    ax.set_title("F  CD8 density", loc="left", fontweight="bold")
    fig.tight_layout()

    table = pd.DataFrame({
        "group": ["COMP_low", "COMP_high"],
        "mean_cd8_density": [float(np.mean(data.cd8_density["COMP_low"])),
                             float(np.mean(data.cd8_density["COMP_high"]))],
    })
    report = ("Synthetic reconstructions of the multiplex immunofluorescence channels "
              "(COMP, a-SMA, FAP, COMP+CD47) and CD8 IHC, plus the quantified CD8+ "
              "density by COMP status.")
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables={"cd8_density": table}, processed={"cd8_density": table},
                       report=report)
