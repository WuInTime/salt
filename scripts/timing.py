#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict, OrderedDict

# Matplotlib is only needed for --mode plot
import matplotlib.pyplot as plt

DATA = {
    "Simulation Time": {
        "Tiled": {
            "3D Tensor Vector": "0:07.25",
            "4D Tensor": "2:52.02",
            "Attention Score": "29:31.60",
            "Batched GEMM": "2:50.64",
            "Context Lookup": "30:53.01",
            "Matrix Matrix": "0:40.00",
            "Matrix Vector": "0:11.10",
            "Rowwise SoftMax": "4:41.11"
        },
        "Constant": {
            "3D Tensor Vector": "0:07.39",
            "4D Tensor": "12:27.31",
            "Attention Score": "59:35.35",
            "Batched GEMM": "8:14.36",
            "Context Lookup": "130:33.00",
            "Matrix Matrix": "3:34.90",
            "Matrix Vector": "0:11.32",
            "Rowwise SoftMax": "4:38.35"
        }
    },
    "Simulation Time (8-way assoc)": {
        "Tiled": {
            "3D Tensor Vector": "0:00.41",
            "4D Tensor": "0:00.72",
            "Attention Score": "0:07.57",
            "Batched GEMM": "0:01.33",
            "Context Lookup": "0:10.06",
            "Matrix Matrix": "0:00.68",
            "Matrix Vector": "0:00.43",
            "Rowwise SoftMax": "0:00.71"
        },
        "Constant": {
            "3D Tensor Vector": "0:00.42",
            "4D Tensor": "0:00.73",
            "Attention Score": "0:07.35",
            "Batched GEMM": "0:01.42",
            "Context Lookup": "0:11.67",
            "Matrix Matrix": "0:00.71",
            "Matrix Vector": "0:00.43",
            "Rowwise SoftMax": "0:00.71"
        }
    },
    "Salt Time": {
        "Tiled": {
        "3D Tensor Vector": "0:00.005257",
        "4D Tensor": "0:00.005670",
        "Attention Score": "0:00.005679",
        "Batched GEMM": "0:00.006115",
        "Context Lookup": "0:00.005433",
        "Matrix Matrix": "0:00.005568",
        "Matrix Vector": "0:00.005698",
        "Rowwise SoftMax": "0:00.005639"
        },
        "Constant": {
        "3D Tensor Vector": "0:00.005752",
        "4D Tensor": "0:00.005629",
        "Attention Score": "0:00.005222",
        "Batched GEMM": "0:00.006149",
        "Context Lookup": "0:00.005837",
        "Matrix Matrix": "0:00.005712",
        "Matrix Vector": "0:00.005319",
        "Rowwise SoftMax": "0:00.005307"
        },
        "Symbolic": {
        "3D Tensor Vector": "0:00.005568",
        "4D Tensor": "0:00.005389",
        "Attention Score": "0:00.006087",
        "Batched GEMM": "0:00.005905",
        "Context Lookup": "0:00.005440",
        "Matrix Matrix": "0:00.005592",
        "Matrix Vector": "0:00.005267",
        "Rowwise SoftMax": "0:00.005422"
        }
    },
    "Barvinok Time": {
        "Tiled": {
            "3D Tensor Vector": "00:01.46",
            "4D Tensor": "00:28.63",
            "Attention Score": "00:26.46",
            "Batched GEMM": "00:08.74",
            "Context Lookup": "00:30.02",
            "Matrix Matrix": "00:01.13",
            "Matrix Vector": "00:00.22",
            "Rowwise SoftMax": "00:02.80"
        },
        "Constant": {
            "3D Tensor Vector": "00:00.53",
            "4D Tensor": "00:01.49",
            "Attention Score": "00:03.03",
            "Batched GEMM": "00:01.31",
            "Context Lookup": "00:03.61",
            "Matrix Matrix": "00:00.58",
            "Matrix Vector": "00:00.21",
            "Rowwise SoftMax": "00:00.79"
        },
        "Symbolic": {
            "3D Tensor Vector": "00:02.02",
            "4D Tensor": "00:06.02",
            "Attention Score": "00:44.01",
            "Batched GEMM": "00:06.02",
            "Context Lookup": "00:37.45",
            "Matrix Matrix": "00:00.59",
            "Matrix Vector": "00:00.46",
            "Rowwise SoftMax": "00:08.25"
        }
    }
}

KERNEL_ORDER = [
    "3D Tensor Vector",
    "4D Tensor",
    "Attention Score",
    "Batched GEMM",
    "Context Lookup",
    "Matrix Matrix",
    "Matrix Vector",
    "Rowwise SoftMax",
]


def format_seconds_sci(s: float) -> str:
    r"""Return a LaTeX-style scientific format like $1.234 \times 10^{3} \,\mathrm{s}$."""
    if s <= 0:
        return r"$0\,\mathrm{s}$"
    e = int(math.floor(math.log10(s)))
    m = s / (10 ** e)
    m_rounded = round(m, 3)
    if m_rounded >= 10:
        m_rounded /= 10
        e += 1
    # Return a mathtext string with single backslashes
    return f"${m_rounded:.3f} \\times 10^{{{e}}}\\,\\mathrm{{s}}$"


def parse_time_to_seconds(s: str) -> float:
    """
    Accepts 'm:ss', 'mm:ss', 'mm:ss.xx', or 'h:mm:ss(.xx)' just in case.
    Returns seconds (float).
    """
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    elif len(parts) == 3:  # support h:mm:ss(.xx)
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    else:
        raise ValueError(f"Unrecognized time format: {s}")


def tidy_rows(data):
    """
    Convert nested dict to a tidy list of rows:
      (family, variant, kernel, seconds, raw_str)
    """
    rows = []
    for family, variants in data.items():
        for variant, kernel_map in variants.items():
            for kernel, raw in kernel_map.items():
                rows.append((family, variant, kernel, parse_time_to_seconds(raw), raw))
    return rows


def families_variants(data):
    """Ordered mapping: family -> list of variants (preserve input order)."""
    fam2vars = OrderedDict()
    for family, variants in data.items():
        fam2vars[family] = list(variants.keys())
    return fam2vars


def plot_all(data, out_path: str, dpi=200):
    rows = tidy_rows(data)
    fam2vars = families_variants(data)

    # Prepare data: family -> variant -> list aligned to KERNEL_ORDER
    fam_variant_kernel_seconds = {}
    fam_variant_kernel_raw = {}
    for family in fam2vars:
        fam_variant_kernel_seconds[family] = {}
        fam_variant_kernel_raw[family] = {}
        for variant in fam2vars[family]:
            vals_sec = []
            vals_raw = []
            for k in KERNEL_ORDER:
                raw = data[family][variant][k]
                vals_raw.append(raw)
                vals_sec.append(parse_time_to_seconds(raw))
            fam_variant_kernel_seconds[family][variant] = vals_sec
            fam_variant_kernel_raw[family][variant] = vals_raw

    # Compute global min/max (seconds) to set shared x-limits with generous headroom for labels
    all_secs = []
    for family in fam_variant_kernel_seconds:
        for var in fam_variant_kernel_seconds[family]:
            all_secs.extend(fam_variant_kernel_seconds[family][var])
    min_sec = min(s for s in all_secs if s > 0)
    max_sec = max(all_secs)
    # Add headroom to the right for text labels; keep left > 0 for log scale
    left_lim = max(min_sec / 1.8, 1e-6)
    right_lim = max_sec * 3.5

    n_families = len(fam2vars)
    fig_height = 2.8 * n_families + 1.2
    fig, axes = plt.subplots(n_families, 1, figsize=(15, fig_height), sharex=True, constrained_layout=True)
    if n_families == 1:
        axes = [axes]

    # Horizontal bar charts with grouped offsets per variant
    for ax, (family, variants) in zip(axes, fam2vars.items()):
        base_y = range(len(KERNEL_ORDER))
        nvar = len(variants)
        offset = 0.20
        start = -offset * (nvar - 1) / 2.0

        y_per_var = []
        x_per_var = []

        for i, var in enumerate(variants):
            y = [y0 + start + i * offset for y0 in base_y]
            x = fam_variant_kernel_seconds[family][var]
            ax.barh(y, x, height=0.12, label=var)
            y_per_var.append(y)
            x_per_var.append(x)

        # All labels outside to the right; vertically stagger within each kernel group
        for idx in range(len(KERNEL_ORDER)):
            entries = []  # (xi, yi)
            for vi in range(nvar):
                xi = x_per_var[vi][idx]
                yi = y_per_var[vi][idx]
                if xi > 0:
                    entries.append((xi, yi))
            if not entries:
                continue
            # Sort by y (top to bottom). In Matplotlib, larger y is higher on the axis.
            entries.sort(key=lambda t: t[1], reverse=True)

            n = len(entries)
            if n == 1:
                dy_list = [0]
            elif n == 2:
                dy_list = [4, -4]  # top up, bottom down
            else:
                dy_list = [4, 0, -4]  # top up, middle center, bottom down

            for (xi, yi), dy in zip(entries, dy_list):
                lbl = format_seconds_sci(xi)
                ax.annotate(lbl, xy=(xi, yi), xytext=(8, dy), textcoords="offset points",
                            va="center", ha="left", fontsize=7.0, color="black", clip_on=False)

        ax.set_yticks(list(base_y))
        ax.set_yticklabels(KERNEL_ORDER, fontweight='bold', fontsize=10)
        ax.set_title(family, fontweight='bold', fontsize=12)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        leg = ax.legend(frameon=False, ncols=min(3, nvar))
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontweight('bold')
                txt.set_fontsize(10)
        ax.set_xscale('log')
        ax.set_xlim(left_lim, right_lim)
        ax.margins(x=0.02)
        # Make tick labels bold
        for tick in ax.get_xticklabels():
            tick.set_fontweight('bold')
            tick.set_fontsize(10)

    axes[-1].set_xlabel("Time (seconds, log scale)", fontweight='bold', fontsize=12)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")


def latex_table(data):
    """
    Emit a LaTeX table (booktabs) with two header rows:
    Families as top-level columns; their variants as subcolumns.
    Values are the original mm:ss.xx strings.
    """
    fam2vars = families_variants(data)

    # Column spec: first column is kernel, then one col per variant of each family
    cols = ["l"]
    for family, variants in fam2vars.items():
        cols.extend(["c"] * len(variants))
    colspec = "".join(cols)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + colspec + r"}")
    lines.append(r"\toprule")

    # First header row: families with \multicolumn
    header_top = [r"\textbf{Kernel}"]
    for family, variants in fam2vars.items():
        header_top.append(r"\multicolumn{" + str(len(variants)) + r"}{c}{\textbf{" + family + r"}}")
    lines.append(" & ".join(header_top) + r" \\")
    lines.append(r"\midrule")

    # Second header row: variants
    header_mid = [""]
    for family, variants in fam2vars.items():
        for v in variants:
            header_mid.append(r"\textit{" + v + r"}")
    lines.append(" & ".join(header_mid) + r" \\")
    lines.append(r"\midrule")

    # Body
    for kernel in KERNEL_ORDER:
        row = [kernel]
        for family, variants in fam2vars.items():
            for v in variants:
                row.append(data[family][v][kernel])
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Timing comparison across kernels and methods.}")
    lines.append(r"\label{tab:kernel-times}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Plot or export LaTeX table for kernel timing data.")
    p.add_argument("--mode", choices=["plot", "latex"], required=True,
                   help="plot: save a multi-panel bar chart; latex: print LaTeX table to stdout.")
    p.add_argument("--out", default="times.png",
                   help="Output path for --mode plot (ignored for --mode latex).")
    args = p.parse_args()

    if args.mode == "plot":
        plot_all(DATA, args.out)
    else:
        latex_table(DATA)


if __name__ == "__main__":
    main()
