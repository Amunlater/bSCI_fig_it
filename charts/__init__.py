"""Chart-type modules for the figIndividual package.

One module per statistical chart type.  Each module exposes ``CHART_TYPE``,
``TITLE``, ``REQUIRED`` (and optionally ``REQUIRED_ANY``) and
``build(data) -> ChartOutput``.

Modules are discovered by ``registry.py`` (see ``CHART_MODULES``).  Chart
modules import the package's shared helpers with absolute imports
(``from config import C`` / ``from io_utils import ...``), so the figIndividual
root must be importable; the bootstrap below guarantees that regardless of how
this subpackage is entered.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
