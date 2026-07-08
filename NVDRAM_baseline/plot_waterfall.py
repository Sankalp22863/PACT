"""
phi-HBM STCO thermal-waterfall figure  (NVDRAM_waterfall)
===========================================================
Renders waterfall_results.csv as a descending staircase of peak compute-die (GPU)
temperature vs the cumulative STCO optimizations -- the phi-HBM analogue of the
paper's Fig. 13, rebuilt from the corrected mold baseline (thermal silicon added
only at its own step).

Single series (phi-HBM), so no legend: the title names it. Direct labels carry the
temperature at each level and the drop on each step; axes/grid are recessive.

Usage:  python3 plot_waterfall.py
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# accent chosen by stack type (paper convention: purple = phi-HBM, red = all-DRAM)
ACCENT_PHI  = "#6a51a3"   # phi-HBM staircase (single-hue, CVD-safe)
ACCENT_DRAM = "#c0392b"   # all-DRAM staircase
DROP  = "#2b8a3e"   # "optimization applied" green pill
INK   = "#2b2b2b"   # primary text
MUTE  = "#8a8a8a"   # secondary text / grid


def read_label():
    path = os.path.join(HERE, "waterfall_meta.txt")
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return "φ-HBM: 4 NVDRAM + 8 DRAM"

SHORT = {
    "0_baseline":     "Baseline\n(3D stacking)",
    "1_base_removal": "+ Base-die\nremoval",
    "2_merge":        "+ Stack\nmerging",
    "3_thin_top":     "+ Top-die\nthinning",
    "4_freq_0p7":     "+ 0.7x GPU\nfrequency",
    "5_thermal_si":   "+ Thermal\nsilicon",
}


def load():
    rows = []
    with open(os.path.join(HERE, "waterfall_results.csv")) as fh:
        for r in csv.DictReader(fh):
            rows.append((r["stage"], r["label"], float(r["peak_gpu_understack_C"])))
    return rows


def main():
    rows = load()
    xs = list(range(len(rows)))
    temps = [t for _, _, t in rows]

    label = read_label()
    PHI = ACCENT_DRAM if "NVDRAM" not in label.upper() else ACCENT_PHI

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    # ── descending staircase: horizontal level at each stage + vertical drop ──
    for i in range(len(rows)):
        ax.plot([i - 0.30, i + 0.30], [temps[i], temps[i]], color=PHI, lw=3.0,
                solid_capstyle="round", zorder=4)
        if i > 0:                                   # vertical connector = the drop
            ax.plot([i - 0.70, i - 0.30], [temps[i - 1], temps[i]], color=PHI,
                    lw=2.0, ls=(0, (1, 1)), zorder=3)
        ax.plot(i, temps[i], "o", ms=9, color=PHI, mec="white", mew=1.5, zorder=5)

    # ── temperature callout above each level ──
    for i, t in enumerate(temps):
        ax.annotate(f"{t:.1f} °C", (i, t), xytext=(0, 12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PHI, lw=1.2))

    # ── drop pill on each step ──
    for i in range(1, len(rows)):
        d = temps[i] - temps[i - 1]
        ymid = (temps[i - 1] + temps[i]) / 2.0
        ax.annotate(f"{d:+.1f}", (i - 0.5, ymid), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.25", fc=DROP, ec="none"), zorder=6)

    # ── total-drop annotation ──
    ax.annotate("", xy=(len(rows) - 1, temps[-1]), xytext=(0, temps[0]),
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0, alpha=0))
    ax.text(0.02, 0.04,
            f"Total: {temps[0]:.1f} → {temps[-1]:.1f} °C   ({temps[-1]-temps[0]:+.1f} °C)",
            transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f0fa", ec=PHI, lw=1.0))

    # ── axes / cosmetics (recessive) ──
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[s] for s, _, _ in rows], fontsize=10, color=INK)
    ax.set_ylabel("Peak compute-die temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(-0.9, len(rows) - 0.4)
    ax.set_ylim(min(temps) - 18, max(temps) + 22)
    ax.grid(axis="y", color=MUTE, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    ax.set_title("STCO Thermal Waterfall — peak compute-die temperature\n"
                 f"{label}; cumulative optimizations from the mold baseline "
                 "(top-side liquid lid; PACT steady-state)",
                 fontsize=13, color=INK, pad=16)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"waterfall.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
