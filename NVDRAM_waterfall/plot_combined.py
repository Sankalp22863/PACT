"""
Combined φ-HBM vs all-DRAM STCO thermal-waterfall figure  (NVDRAM_waterfall)
==============================================================================
Overlays the two calibrated waterfalls on one axes -- the φ-HBM (hybrid, this
folder) staircase against the all-DRAM (../NVDRAM_baseline) staircase -- the
direct analogue of the paper's Fig. 13. Reads both waterfall_results.csv files.

Output: combined_waterfall.png / .pdf in this folder.

Usage:  python3 plot_combined.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

PHI  = "#6a51a3"   # φ-HBM (hybrid)
DRAM = "#c0392b"   # all-DRAM
DROP = "#2b8a3e"
INK  = "#2b2b2b"
MUTE = "#8a8a8a"

SHORT = ["Baseline\n(3D stacking)", "+ Base-die\nremoval", "+ Stack\nmerging",
         "+ Top-die\nthinning", "+ 0.7x GPU\nfrequency", "+ Thermal\nsilicon"]


def load(path):
    with open(path) as fh:
        return [float(r["peak_gpu_understack_C"]) for r in csv.DictReader(fh)]


def staircase(ax, temps, color, tlabel_dy, name):
    """Draw one descending staircase with level segments + dotted drops."""
    n = len(temps)
    for i in range(n):
        lbl = name if i == 0 else None
        ax.plot([i - 0.30, i + 0.30], [temps[i], temps[i]], color=color, lw=3.0,
                solid_capstyle="round", zorder=4, label=lbl)
        if i > 0:
            ax.plot([i - 0.70, i - 0.30], [temps[i - 1], temps[i]], color=color,
                    lw=2.0, ls=(0, (1, 1)), zorder=3)
        ax.plot(i, temps[i], "o", ms=8, color=color, mec="white", mew=1.4, zorder=5)
        ax.annotate(f"{temps[i]:.1f}", (i, temps[i]), xytext=(0, tlabel_dy),
                    textcoords="offset points", ha="center",
                    va="bottom" if tlabel_dy > 0 else "top",
                    fontsize=9.5, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=1.0))


def main():
    phi  = load(os.path.join(HERE, "waterfall_results.csv"))
    dram = load(os.path.join(HERE, "..", "NVDRAM_baseline", "waterfall_results.csv"))
    n = len(phi)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    # shade the NVDRAM benefit (gap between the two staircases)
    ax.fill_between(xs, phi, dram, step=None, color=PHI, alpha=0.07, zorder=1)

    staircase(ax, dram, DRAM, +12, "All-DRAM: 12 DRAM")     # hotter -> labels above
    staircase(ax, phi,  PHI,  -14, "φ-HBM: 4 NVDRAM + 8 DRAM")  # cooler -> labels below

    # per-stage NVDRAM benefit (Δ) between the curves, centred in the gap
    for i in range(n):
        d = phi[i] - dram[i]
        ax.annotate(f"{d:+.1f}", (i, (phi[i] + dram[i]) / 2), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", fc=DROP, ec="none"), zorder=6)

    ax.text(0.015, 0.05,
            f"All-DRAM {dram[0]:.1f}→{dram[-1]:.1f} °C   |   "
            f"φ-HBM {phi[0]:.1f}→{phi[-1]:.1f} °C   |   "
            f"NVDRAM benefit ≈ {abs(sum(phi[i]-dram[i] for i in range(n))/n):.1f} °C/stage",
            transform=ax.transAxes, fontsize=10.5, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f5fb", ec=MUTE, lw=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels(SHORT, fontsize=10, color=INK)
    ax.set_ylabel("Peak compute-die temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(-0.9, n - 0.35)
    ax.set_ylim(min(phi) - 16, max(dram) + 20)
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

    ax.set_title("STCO Thermal Waterfall — φ-HBM vs all-DRAM\n"
                 "calibrated to the reference baseline (all-DRAM 3D = 141.4 °C ≈ reference 141.7 °C); "
                 "green = NVDRAM benefit per stage",
                 fontsize=13, color=INK, pad=16)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"combined_waterfall.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
