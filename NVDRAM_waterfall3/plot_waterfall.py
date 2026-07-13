"""
All-DRAM STCO waterfall figure (30x22 die) — GPU + bottom-DRAM  (NVDRAM_baseline2)
=================================================================================
Descending staircase of peak compute-die (GPU) temperature vs the cumulative STCO
optimizations, with the BOTTOM-MOST DRAM tier temperature overlaid as a second
series at every stage. Reads waterfall_results.csv.

Output: waterfall.png / .pdf
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

GPU_C  = "#6a51a3"   # GPU compute die (φ-HBM purple)
DRAM_C = "#2c7fb8"   # lowest DRAM tier (steel blue)
DROP   = "#2b8a3e"
INK    = "#2b2b2b"
MUTE   = "#8a8a8a"

SHORT = ["Baseline\n(3D stacking)", "+ Base-die\nremoval", "+ Stack\nmerging",
         "+ Top-die\nthinning", "+ 0.7x GPU\nfrequency", "+ Thermal\nsilicon"]


def load():
    gpu, dram = [], []
    with open(os.path.join(HERE, "waterfall_results.csv")) as fh:
        for r in csv.DictReader(fh):
            gpu.append(float(r["peak_gpu_C"]))
            dram.append(float(r["bottom_dram_C"]))
    return gpu, dram


def staircase(ax, temps, color, dy, name, pills=False):
    n = len(temps)
    for i in range(n):
        ax.plot([i - 0.30, i + 0.30], [temps[i], temps[i]], color=color, lw=3.0,
                solid_capstyle="round", zorder=4, label=name if i == 0 else None)
        if i > 0:
            ax.plot([i - 0.70, i - 0.30], [temps[i - 1], temps[i]], color=color,
                    lw=2.0, ls=(0, (1, 1)), zorder=3)
        ax.plot(i, temps[i], "o", ms=8, color=color, mec="white", mew=1.4, zorder=5)
        ax.annotate(f"{temps[i]:.1f}", (i, temps[i]), xytext=(0, dy),
                    textcoords="offset points", ha="center",
                    va="bottom" if dy > 0 else "top", fontsize=9.5, fontweight="bold",
                    color=INK, bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=1.0))
        if pills and i > 0:
            d = temps[i] - temps[i - 1]
            ax.annotate(f"{d:+.1f}", (i - 0.5, (temps[i - 1] + temps[i]) / 2), ha="center",
                        va="center", fontsize=8.5, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=DROP, ec="none"), zorder=6)


def main():
    gpu, dram = load()
    n = len(gpu)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.fill_between(xs, dram, gpu, color=GPU_C, alpha=0.06, zorder=1)
    staircase(ax, gpu,  GPU_C,  +12, "GPU compute die (peak)", pills=True)
    staircase(ax, dram, DRAM_C, -14, "Lowest DRAM tier (peak)")

    ax.text(0.015, 0.05,
            f"GPU {gpu[0]:.1f}→{gpu[-1]:.1f} °C   |   lowest-DRAM {dram[0]:.1f}→{dram[-1]:.1f} °C   |   "
            f"φ-HBM (4 NVDRAM + 8 DRAM), 30×22 mm",
            transform=ax.transAxes, fontsize=10.5, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f5fb", ec=MUTE, lw=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT, fontsize=10, color=INK)
    ax.set_ylabel("Peak temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(-0.9, n - 0.35)
    ax.set_ylim(min(dram) - 16, max(gpu) + 20)
    ax.grid(axis="y", color=MUTE, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)
    ax.legend(loc="upper right", frameon=True, fontsize=11, framealpha=0.95)

    ax.set_title("φ-HBM STCO Thermal Waterfall — 30 × 22 mm die, active read/write\n"
                 "GPU compute-die peak + lowest DRAM tier; 4 NVDRAM + 8 DRAM, 355 W GPU; "
                 "memory +active R/W (α$_w$=0.24, 4.4 TB/s/stack, 90/70 fJ/bit)",
                 fontsize=12.5, color=INK, pad=16)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"waterfall.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
