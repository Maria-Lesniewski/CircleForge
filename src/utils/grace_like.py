#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_xy(filename):
    data = np.loadtxt(filename)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{filename}: expected at least two columns (x y)")

    return data[:, 0], data[:, 1]


def main():
    parser = argparse.ArgumentParser(
        description="Plot one or more XY data files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="XY data files (whitespace-separated x y columns)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Plot title",
    )
    parser.add_argument(
        "--xlabel",
        default="x",
        help="X-axis label",
    )
    parser.add_argument(
        "--ylabel",
        default="y",
        help="Y-axis label",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Use logarithmic x-axis",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic y-axis",
    )
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9, 6))

    for filename in args.files:
        x, y = load_xy(filename)

        # Just the filename, not the full path.
        label = Path(filename).name

        ax.scatter(x, y, label=label)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)

    if args.title:
        ax.set_title(args.title)

    if args.logx:
        ax.set_xscale("log")

    if args.logy:
        ax.set_yscale("log")

    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

