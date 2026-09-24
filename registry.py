"""Chart registry.

Maps a chart-type name to its module.  Each module exposes ``CHART_TYPE``,
``TITLE``, ``REQUIRED`` and ``build(data) -> ChartOutput``.  Adding a new chart
type only requires dropping a module into ``charts/`` and listing it below.
"""
import os
import sys
import importlib

# Make the figIndividual root importable (so ``charts`` resolves when this
# module is imported from another working directory).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# One module per statistical chart type. Order is only the display order.
CHART_MODULES = [
    "charts.chart_survival",
    "charts.chart_roc",
    "charts.chart_forest",
    "charts.chart_lasso",
    "charts.chart_calibration",
    "charts.chart_dca",
    "charts.chart_nomogram",
    "charts.chart_dimred",
    "charts.chart_heatmap",
    "charts.chart_bar",
    "charts.chart_box",
    "charts.chart_violin",
    "charts.chart_scatter",
    "charts.chart_network",
    "charts.chart_image",
    "charts.chart_table",
]


def load_registry():
    registry = {}
    for name in CHART_MODULES:
        module = importlib.import_module(name)
        registry[module.CHART_TYPE] = module
    return registry


REGISTRY = load_registry()
CHART_TYPES = sorted(REGISTRY)


def describe():
    """Human-readable list of available chart types."""
    return [(t, REGISTRY[t].TITLE, ", ".join(getattr(REGISTRY[t], "REQUIRED", ())) or "-")
            for t in CHART_TYPES]
