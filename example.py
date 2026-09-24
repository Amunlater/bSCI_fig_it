"""Example / demo for the figIndividual package.

Runs the whole pipeline on randomly generated data so the effect of every chart
module can be inspected without any real dataset.  Also shows how to drive a
single chart module directly.

    python example.py
    python example.py --out F:\\some\\folder --charts survival,roc,bar
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulate import simulate_paper_data          # noqa: E402
from pipeline import run, available_charts         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description="figIndividual random-data demo")
    ap.add_argument("--out", default=os.path.join(HERE, "output_demo"),
                    help="output folder for the demo (default: figIndividual/output_demo)")
    ap.add_argument("--charts", default=None,
                    help="comma-separated chart types (default: all)")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    args = ap.parse_args(argv)

    charts = None
    if args.charts:
        charts = [c.strip() for c in args.charts.split(",") if c.strip()]

    print("=" * 68)
    print("figIndividual demo - randomly generated paper-like data")
    print("=" * 68)

    # Build a fully-populated random dataset explicitly (so the example is
    # self-documenting about the data contract).
    data = simulate_paper_data(n=500, seed=args.seed)
    print(f"[demo] generated PaperData with {len(data.filled())} populated fields")
    print(f"[demo] charts supported by this data: {available_charts(data)}")

    # Run through the same pipeline the CLI uses.
    outputs = run(out_dir=args.out, charts=charts, demo=True, seed=args.seed)

    print("\n[demo] wrote:")
    for out in outputs:
        print(f"  - {out.chart_type:12s} {out.title}")
    print(f"\n[demo] inspect the results under: {os.path.abspath(args.out)}")

    # -- how to drive a single chart module directly --------------------
    print("\n[demo] single-chart example (forest only):")
    from io_utils import OutputManager
    import charts.chart_forest as chart_forest
    out = chart_forest.build(data)
    mgr = OutputManager(os.path.join(args.out, "_single"))
    mgr.write(out)
    print(f"[demo] wrote figures/{out.chart_type}.png via OutputManager")


if __name__ == "__main__":
    main()
