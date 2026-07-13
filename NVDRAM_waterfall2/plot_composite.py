"""
Composite STCO waterfall — φ-HBM vs all-DRAM  (NVDRAM_waterfall2)
================================================================
GPU compute-die peak plotted as a step/staircase (same style as the individual
waterfalls), for both experiments:
  * all-DRAM (../NVDRAM_baseline2)  — red
  * φ-HBM    (this folder)          — blue
At each stage the GPU temperature is written as a number on its line, and the
corresponding lowest-DRAM temperature is shown in a filled box directly ABOVE that
experiment's GPU number. Output: composite_waterfall.png / .pdf
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTE = "#2b2b2b", "#8a8a8a"
AD, PH = "#c0392b", "#2c7fb8"          # all-DRAM red, φ-HBM blue

STAGES = ["Baseline\n(3D stacking)", "+ Base-die\nremoval", "+ Stack\nmerging",
          "+ Top-die\nthinning", "+ Freq\nscaling", "+ Thermal\nsilicon"]


def load(path):
    gpu, dram = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            gpu.append(float(r["peak_gpu_C"]))
            dram.append(float(r["bottom_dram_C"]))
    return gpu, dram


def staircase(ax, ys, color):
    """GPU step line: flat level segment per stage + dotted diagonal drops + circles
    (same style as the individual waterfall plots)."""
    n = len(ys)
    for i in range(n):
        ax.plot([i - 0.30, i + 0.30], [ys[i], ys[i]], color=color, lw=3.0,
                solid_capstyle="round", zorder=5)
        if i > 0:
            ax.plot([i - 0.70, i - 0.30], [ys[i - 1], ys[i]], color=color,
                    lw=2.0, ls=(0, (1, 1)), zorder=4)
        ax.plot(i, ys[i], "o", ms=8, color=color, mec="white", mew=1.4, zorder=6)


def annotate(ax, gpu, dram, color, above):
    """GPU temp as a number by the line; DRAM temp in a filled box above the GPU
    number. Top line labels go ABOVE the line, bottom line labels go BELOW it (so
    the two experiments' labels never overlap). DRAM stays above the GPU number."""
    if above:                       # top line: line -> GPU number -> DRAM box (upward)
        gpu_dy, dram_dy, va = 10, 27, "bottom"
    else:                           # bottom line: line -> GPU number -> DRAM box (downward)
        gpu_dy, dram_dy, va = -10, -27, "top"
    for i in range(len(gpu)):
        ax.annotate(f"{gpu[i]:.1f}", (i, gpu[i]), xytext=(0, gpu_dy), textcoords="offset points",
                    ha="center", va=va, fontsize=9, fontweight="bold", color=color, zorder=7)
        ax.annotate(f"{dram[i]:.1f}", (i, gpu[i]), xytext=(0, dram_dy), textcoords="offset points",
                    ha="center", va=va, fontsize=8.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc=color, ec=color, lw=1.0), zorder=7)


def main():
    ad_gpu, ad_dram = load(os.path.join(HERE, "..", "NVDRAM_baseline2", "waterfall_results.csv"))
    ph_gpu, ph_dram = load(os.path.join(HERE, "waterfall_results.csv"))
    n = len(ad_gpu)

    fig, ax = plt.subplots(figsize=(11.5, 8.4))

    staircase(ax, ad_gpu, AD)
    staircase(ax, ph_gpu, PH)
    annotate(ax, ad_gpu, ad_dram, AD, above=True)    # all-DRAM = top line -> labels above
    annotate(ax, ph_gpu, ph_dram, PH, above=False)   # φ-HBM = bottom line -> labels below

    handles = [
        Line2D([0], [0], color=AD, lw=3.0, marker="o", mec="white", label="All-DRAM — GPU"),
        Line2D([0], [0], color=PH, lw=3.0, marker="o", mec="white", label="φ-HBM — GPU"),
        Line2D([0], [0], color=MUTE, lw=0, marker="s", ms=11, mfc="#888", mec="none",
               label="filled box = lowest DRAM tier"),
    ]

    ax.set_xticks(range(n))
    ax.set_xticklabels(STAGES, fontsize=10, color=INK)
    ax.set_ylabel("GPU peak temperature (°C)   ·   box = lowest DRAM tier", fontsize=11.5, color=INK)
    ax.set_xlim(-0.75, n - 0.25)
    ax.set_ylim(min(ph_gpu) - 12, max(ad_gpu) + 14)  # room for labels above top / below bottom line
    ax.grid(axis="y", color=MUTE, alpha=0.3, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"composite_waterfall.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
