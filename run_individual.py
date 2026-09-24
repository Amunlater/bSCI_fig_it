"""Unified command-line entry for the figIndividual chart package.

Usage
-----
    python run_individual.py <input_path> [--output OUT] [--charts TYPES]

``<input_path>`` is a folder holding a dataset: GEO / TCGA downloads, an
expression matrix, sample/clinical tables, annotation files, or raw sequencing
data.  The folder contents are auto-detected and only the charts the data can
support are drawn (unless ``--charts`` forces a set).

Examples
--------
    # auto-detect a GEO / TCGA folder, auto-select charts
    python run_individual.py F:\\RProject\\GSE\\GSE222315_RAW --output F:\\RProject\\figures\\figIndividual\\output

    # only survival + roc + bar charts
    python run_individual.py D:\\data\\TCGA_BLCA --charts survival,roc,bar

    # everything, filling missing fields with simulated values
    python run_individual.py D:\\data\\GSE39582 --charts all

    # no dataset -> simulated paper statistics
    python run_individual.py --demo

    # list chart types
    python run_individual.py --list-charts
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run, list_charts, CHART_TYPES  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="run_individual",
        description="Draw one statistical chart type per module from a dataset folder.")
    ap.add_argument("input", nargs="?", default=None,
                    help="input file path (folder containing a dataset, expression "
                         "matrix, clinical table, annotation, or raw sequencing data)")
    ap.add_argument("-o", "--output", default=None,
                    help="output root directory (default: figIndividual/output)")
    ap.add_argument("-c", "--charts", default=None,
                    help="comma-separated chart types, or 'all' "
                         f"(available: {', '.join(CHART_TYPES)})")
    ap.add_argument("--demo", action="store_true",
                    help="force simulated paper data (ignore input)")
    ap.add_argument("--n-cap", type=int, default=2500,
                    help="max cells per sample for single-cell loading (default 2500)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    ap.add_argument("--force", action="store_true", help="rebuild cached single-cell data")
    ap.add_argument("--no-fill", action="store_true",
                    help="do not fill missing data fields with simulated values")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                    help="figure format (default png)")
    ap.add_argument("--list-charts", action="store_true", help="list chart types and exit")
    args = ap.parse_args(argv)

    if args.list_charts:
        list_charts()
        return

    charts = None
    if args.charts and args.charts.strip().lower() != "all":
        charts = [c.strip() for c in args.charts.split(",") if c.strip()]

    run(input_path=args.input, out_dir=args.output, charts=charts, demo=args.demo,
        n_cap=args.n_cap, seed=args.seed, force=args.force,
        fill_missing=not args.no_fill, fmt=args.format)


if __name__ == "__main__":
    main()
