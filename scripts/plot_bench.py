#!/usr/bin/env python3
"""Plot latency and bandwidth vs message size from a bench CSV.

    python3 scripts/plot_bench.py results/bench_np4.csv -o results/bench_np4.png

Bandwidth is the algorithm bandwidth: message size / time per call.
"""
import argparse
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [("native_us", "MPI_Allreduce"), ("tree_us", "tree"), ("ring_us", "ring")]


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path}: no data")
    data = {"bytes": [int(r["bytes"]) for r in rows]}
    for key, _ in SERIES:
        data[key] = [float(r[key]) for r in rows]
    return data


def size_label(b):
    for unit, scale in (("MB", 1 << 20), ("KB", 1 << 10)):
        if b >= scale:
            return f"{b // scale} {unit}"
    return f"{b} B"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("-o", "--output", default=None, help="output image (default: <csv>.png)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    d = read_csv(args.csv)
    out = args.output or args.csv.rsplit(".", 1)[0] + ".png"
    x = d["bytes"]

    fig, (ax_lat, ax_bw) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in SERIES:
        t = d[key]
        ax_lat.plot(x, t, marker="o", ms=3, label=label)
        ax_bw.plot(x, [b / (us * 1e3) for b, us in zip(x, t)], marker="o", ms=3, label=label)

    ticks = x[::3]
    for ax in (ax_lat, ax_bw):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([size_label(b) for b in ticks], rotation=45)
        ax.set_xlabel("message size")
        ax.grid(True, which="major", alpha=0.3)
        ax.legend()

    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("time per call (us)")
    ax_lat.set_title("Latency")
    ax_bw.set_yscale("log")
    ax_bw.set_ylabel("bandwidth (GB/s)")
    ax_bw.set_title("Algorithm bandwidth")

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
