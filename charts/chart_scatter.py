"""Scatter plots.

Panel A: patients ranked by risk score, with survival events marked.  Panels
B-D: spatial distribution of COMP, CD8A and COMP-high spots.  Panel E: spatial
distance-decay of immune marker expression away from COMP-high regions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import C
from io_utils import ChartOutput, subplot_grid

CHART_TYPE = "scatter"
TITLE = "Risk-score and spatial scatter plots"
REQUIRED = ("risk_scores",)


def _spatial(ax, coords, values, cmap, title, vmin=0, vmax=1):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=15,
                    alpha=0.8, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, shrink=0.6)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title, loc="left", fontweight="bold")


def build(data):
    panels = []
    tables, processed = {}, {}

    rs = np.asarray(data.risk_scores, float)
    order = np.argsort(rs)
    groups = np.asarray(data.risk_groups) if data.risk_groups is not None else None
    events = np.asarray(data.survival_events).astype(bool) if data.survival_events is not None else None

    def draw_risk(ax):
        cols = [C["high"] if (groups is not None and g == "High") else C["low"]
                for g in (groups if groups is not None else ["Low"] * len(rs))]
        ax.scatter(range(len(rs)), rs[order], c=cols, s=8, alpha=0.7)
        ax.axhline(np.median(rs), color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Patients (ranked)")
        ax.set_ylabel("Risk Score")
        ax.set_title("A  Risk score distribution", loc="left", fontweight="bold")
        if events is not None:
            ev = events[order]
            axb = ax.twinx()
            axb.scatter(np.where(~ev)[0], np.ones((~ev).sum()) * -0.5, c=C["low"], s=4, alpha=0.5)
            axb.scatter(np.where(ev)[0], np.ones(ev.sum()) * -0.5, c=C["high"], s=4,
                        alpha=0.5, marker="x")
            axb.set_ylim(-1, 1)
            axb.set_yticks([])

    panels.append(("A  Risk score distribution", draw_risk))

    coords = data.spatial_coords
    if coords is not None and data.spatial_comp is not None:
        def draw_comp(ax):
            _spatial(ax, coords, data.spatial_comp, "Oranges", "B  COMP spatial")
        panels.append(("B  COMP spatial", draw_comp))

        if data.spatial_cd8a is not None:
            def draw_cd8(ax):
                _spatial(ax, coords, data.spatial_cd8a, "Blues", "C  CD8A spatial")
            panels.append(("C  CD8A spatial", draw_cd8))

        def draw_high(ax):
            high = data.spatial_comp > 0.5
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=np.where(high, C["high"], C["grey"]), s=15, alpha=0.6)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("D  COMP-high regions", loc="left", fontweight="bold")
            ax.legend([Patch(color=C["high"]), Patch(color=C["grey"])],
                      ["COMP-high", "COMP-low"], fontsize=8, loc="upper right")
        panels.append(("D  COMP-high regions", draw_high))

    if data.distance_decay is not None:
        def draw_dd(ax):
            db = data.distance_decay["dist_bins"]
            ax.plot(db + 25, data.distance_decay["CD8A"], "o-", color=C["high"], lw=2, label="CD8A")
            if "CD3D" in data.distance_decay:
                ax.plot(db + 25, data.distance_decay["CD3D"], "s--", color=C["low"], lw=2, label="CD3D")
            ax.set_xlabel("Distance from COMP-high (um)")
            ax.set_ylabel("Mean Expression")
            ax.set_title("E  Distance-decay", loc="left", fontweight="bold")
            ax.legend(fontsize=8)
        panels.append(("E  Distance-decay", draw_dd))

    fig, axes = subplot_grid(len(panels), ncols=3, panel_size=(6.5, 5.5))
    for ax, (_, draw) in zip(axes, panels):
        draw(ax)
    fig.tight_layout()

    tables["risk_scores"] = pd.DataFrame({
        "patient_rank": np.arange(len(rs)),
        "risk_score": rs[order],
        "group": (groups[order] if groups is not None else "NA"),
    })
    processed["risk_scores"] = tables["risk_scores"]
    if coords is not None:
        spatial = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1],
            "COMP": data.spatial_comp if data.spatial_comp is not None else np.nan,
            "CD8A": data.spatial_cd8a if data.spatial_cd8a is not None else np.nan,
        })
        tables["spatial"] = spatial
        processed["spatial"] = spatial
    if data.distance_decay is not None:
        dd = pd.DataFrame(data.distance_decay)
        tables["distance_decay"] = dd
        processed["distance_decay"] = dd

    report = (
        "Panel A ranks patients by risk score with death events marked below. "
        "Panels B-D map COMP / CD8A / COMP-high spots in space. Panel E shows the "
        "spatial decay of immune expression with distance from COMP-high regions."
    )
    return ChartOutput(chart_type=CHART_TYPE, title=TITLE, figure=fig,
                       tables=tables, processed=processed, report=report)
