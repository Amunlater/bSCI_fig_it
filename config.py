"""Shared configuration for the figIndividual chart package.

Holds the colour palette, matplotlib defaults and the canonical output layout.
Every chart module imports its colours from here so the whole set stays visual
consistent regardless of which chart type is requested.
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, "output")

# Standard output sub-directories produced by the OutputManager.
OUTPUT_SUBDIRS = ("figures", "tables", "processed", "reports")

# Shared semantic colour palette (kept identical to the original figure code).
C = {
    "high": "#E64B35",
    "low": "#4DBBD5",
    "comp": "#00A087",
    "mm3": "#E64B35",
    "mm1": "#F39B7F",
    "ephx4": "#8491B4",
    "plcd4": "#91D1C2",
    "sig": "#3C5488",
    "grey": "#BABABA",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})
