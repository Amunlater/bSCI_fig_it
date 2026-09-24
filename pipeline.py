"""Pipeline orchestrator.

Single place that ties everything together:

    input path  ->  dataset_loader  ->  PaperData  ->  chart modules  ->  OutputManager

The pipeline decides *which* charts to run.  With ``charts=None`` it runs every
chart whose required data fields are present (auto mode).  With an explicit list
it runs exactly those, filling any missing fields with simulated values unless
``fill_missing=False``.
"""
import os
import datetime as _dt

from config import DEFAULT_OUT_DIR
from data_interface import PaperData
from simulate import simulate_paper_data, merge_with_simulated
from io_utils import OutputManager
from registry import REGISTRY, CHART_TYPES


def available_charts(data, registry=None):
    """Chart types whose data requirements are met.

    A chart declares ``REQUIRED`` (all must be present) and optionally
    ``REQUIRED_ANY`` (at least one must be present).
    """
    reg = registry or REGISTRY
    result = []
    for ctype, module in reg.items():
        required = getattr(module, "REQUIRED", ())
        required_any = getattr(module, "REQUIRED_ANY", ())
        all_ok = all(getattr(data, a, None) is not None for a in required)
        any_ok = (not required_any) or any(getattr(data, a, None) is not None for a in required_any)
        if all_ok and any_ok:
            result.append(ctype)
    return sorted(result)


def run(input_path=None, out_dir=None, charts=None, demo=False,
        n_cap=2500, seed=42, force=False, fill_missing=True, fmt="png"):
    """Run the full figure pipeline.

    Parameters
    ----------
    input_path : str, optional
        Folder containing a dataset (GEO / TCGA / expression matrix / clinical
        table / raw single-cell counts).  Omit for simulated data.
    out_dir : str, optional
        Unified output root. Defaults to ``figIndividual/output``.
    charts : list[str], optional
        Explicit chart types to render. ``None`` = auto-select.
    demo : bool
        Force simulated paper data (ignore ``input_path``).
    n_cap, seed, force : scRNA subsampling cap, RNG seed, rebuild cache.
    fill_missing : bool
        Fill data fields that a requested chart needs but the dataset lacks.
    fmt : str
        Figure format (``png`` / ``pdf`` / ``svg``).

    Returns
    -------
    list[ChartOutput]
    """
    started = _dt.datetime.now()
    info = {"run_started": started.strftime("%Y-%m-%d %H:%M:%S")}

    # -- resolve data ---------------------------------------------------
    if input_path and not demo:
        from dataset_loader import load_inputs
        data, load_info = load_inputs(input_path, n_cap=n_cap, seed=seed, force=force)
        info.update(load_info)
    else:
        data = simulate_paper_data(seed=seed)
        info["source"] = "simulated paper statistics" if not input_path else "demo (simulated)"

    # -- decide charts ---------------------------------------------------
    unknown = []
    if charts:
        wanted = []
        for c in charts:
            if c in REGISTRY:
                wanted.append(c)
            else:
                unknown.append(c)
        if fill_missing:
            data = merge_with_simulated(data, seed=seed)
        selected = wanted
    else:
        selected = available_charts(data)
        if not selected and fill_missing:
            print("[pipeline] dataset too sparse for auto mode -> filling with simulated data")
            data = merge_with_simulated(data, seed=seed)
            selected = available_charts(data)

    if unknown:
        print(f"[pipeline] unknown chart type(s): {unknown} (available: {CHART_TYPES})")
    if not selected:
        print("[pipeline] nothing to draw")
        return []

    info["charts_requested"] = ", ".join(selected)

    # -- write -----------------------------------------------------------
    out_dir = os.path.abspath(out_dir or DEFAULT_OUT_DIR)
    manager = OutputManager(out_dir, fmt=fmt)
    outputs = []
    for ctype in selected:
        module = REGISTRY[ctype]
        try:
            out = module.build(data)
        except Exception as exc:  # keep one failing chart from killing the run
            print(f"[pipeline] chart '{ctype}' failed: {type(exc).__name__}: {exc}")
            continue
        manager.write(out)
        outputs.append(out)
        print(f"[pipeline] {ctype:12s} -> figure + {len(out.tables)} table(s) "
              f"+ {len(out.processed)} processed file(s)")

    summary = manager.write_summary(outputs, input_info=info, data=data)
    elapsed = (_dt.datetime.now() - started).total_seconds()
    print(f"\n[pipeline] {len(outputs)} chart(s) written to {out_dir} in {elapsed:.1f}s")
    print(f"[pipeline] summary report: {summary}")
    return outputs


def list_charts():
    """Print the available chart types."""
    from registry import describe
    print(f"{'type':12s}  {'required PaperData fields'}")
    print("-" * 70)
    for ctype, title, req in describe():
        print(f"{ctype:12s}  {req}")
        print(f"{'':12s}  {title}")
