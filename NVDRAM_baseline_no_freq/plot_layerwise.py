"""
Layerwise stack temperature profile — all-DRAM, no frequency scaling
====================================================================
Peak temperature of every DRAM die vs its position in the stack (tier 1 = bottom,
nearest the GPU; tier 12 = top, nearest the lid) for THIS experiment
(NVDRAM_baseline_no_freq — all-DRAM, 414 W, no GPU frequency scaling). Same style
as the layerwise plots in ../NVDRAM_waterfall3 (blue dashed line, square markers =
the all-DRAM stack). Each die temperature is the peak under the stack columns
(same mask as the waterfall). One figure per stage:
  * 0_baseline    (3D stacking, no optimizations) -> layerwise_baseline.png/.pdf
  * 4_thermal_si  (after thermal-silicon opt.)     -> layerwise_thermal_si.png/.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005    # under-stack mask geometry (matches run_waterfall)

# ---- color scheme (matches the "Pure DRAM stack" style from NVDRAM_waterfall3) ----
BLUE = "#2c7fb8"        # all-DRAM stack (dashed line + square markers) + its annotation
INK, MUTE = "#2b2b2b", "#8a8a8a"

STAGES = [
    ("0_baseline",   "3D-stacking baseline (no optimizations)", "layerwise_baseline"),
    ("4_thermal_si", "after thermal-silicon optimization",      "layerwise_thermal_si"),
]


def stack_mask():
    gl = GPU_X / COLS
    cw = (GPU_X - 2 * EDGE_INSERT - CENTRAL_VOID) / 2.0
    xs = [(EDGE_INSERT, EDGE_INSERT + cw),
          (EDGE_INSERT + cw + CENTRAL_VOID, EDGE_INSERT + cw + CENTRAL_VOID + cw)]
    m = np.zeros((ROWS, COLS), dtype=bool)
    for x0, x1 in xs:
        m[:, round(x0 / gl):round(x1 / gl)] = True
    return m


MASK = stack_mask()


def grid(stage, layer):
    g = np.loadtxt(os.path.join(HERE, stage, f"{PREFIX}.layer{layer}")) - 273.15
    return g.reshape(ROWS, COLS)


def profile(stage):
    """[(tier, peak_C), ...] for the DRAM dies, bottom (nearest GPU) -> top.
    Layer indices are read from THIS stage's LCF (base-die presence shifts them)."""
    lcf = os.path.join(HERE, stage, "hybrid_lcf.csv")
    mem = []
    with open(lcf) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "dram_tier_flp.csv":
                mem.append(int(idx))
    mem.sort()
    return [(pos + 1, float(grid(stage, lyr)[MASK].max())) for pos, lyr in enumerate(mem)]


def make(stage, out_name):
    dram = profile(stage)
    n = len(dram)
    xs = [p for p, _ in dram]
    ys = [t for _, t in dram]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    ax.plot(xs, ys, "--", color=BLUE, lw=2.2, marker="s", ms=8, mec="white", mew=1.0, zorder=5)

    # tier-1 (bottom, hottest die) peak callout
    ax.annotate(f"{ys[0]:.1f} °C", (1, ys[0]), xytext=(6, 11), textcoords="offset points",
                ha="left", va="bottom", fontsize=10, fontweight="bold", color=BLUE, zorder=7)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([str(i) for i in range(1, n + 1)], fontsize=9.5, color=INK)
    ax.set_xlabel("Memory tier index   (1 = bottom, nearest GPU  →  top = lid)", fontsize=11, color=INK)
    ax.set_ylabel("Peak die temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylim(min(ys) - 3, max(ys) + 6)
    ax.grid(True, color=MUTE, alpha=0.25, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    handles = [Line2D([0], [0], color=BLUE, lw=2.2, ls="--", marker="s", ms=8, mec="white",
                      label="All-DRAM stack (no freq scaling)")]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9.5, framealpha=0.95)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{out_name}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


def main():
    for stage, _title, out_name in STAGES:
        make(stage, out_name)


if __name__ == "__main__":
    main()
