"""
Layerwise stack temperature profile — all-DRAM vs φ-HBM
=======================================================
Peak temperature of every memory die vs its position in the stack (tier 1 =
bottom, nearest the GPU; tier 12 = top, nearest the lid), for:
  * Pure DRAM stack (HBM only)  — blue dashed, square markers
  * Fe-RAM + DRAM hybrid stack  = this folder (φ-HBM)   — purple line; Fe-RAM tiers
    as red circles, DRAM tiers as orange squares; Fe-RAM region shaded.
Each die temperature is the peak under the stack columns (same mask as the
waterfall). One figure per stage:
  * 0_baseline    -> layerwise_baseline.png/.pdf   — Pure-DRAM line from ../NVDRAM_baseline3
  * 5_thermal_si  -> layerwise_thermal_si.png/.pdf — Pure-DRAM line from
                     ../NVDRAM_baseline_no_freq (4_thermal_si, no frequency scaling)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE3 = os.path.join(HERE, "..", "NVDRAM_baseline3")            # pure-DRAM stack (with freq scaling)
BASE_NOFREQ = os.path.join(HERE, "..", "NVDRAM_baseline_no_freq")  # pure-DRAM stack (no freq scaling)
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005    # under-stack mask geometry (matches run_waterfall)

# ---- color scheme (per the reference mockup) ----
PURPLE = "#7d3c98"      # Fe-RAM + DRAM hybrid stack line + its annotation
NV_RED = "#c0392b"      # Fe-RAM tier markers (circles) + Fe-RAM region shading/label
DR_ORG = "#e08214"      # DRAM tier markers (squares) on the hybrid stack
BLUE = "#2c7fb8"        # pure DRAM stack (dashed line + square markers) + its annotation
INK, MUTE = "#2b2b2b", "#8a8a8a"

# (hyb_stage, pure-DRAM dir, pure-DRAM stage, title, out_name)
# The thermal-silicon figure takes its Pure-DRAM line from NVDRAM_baseline_no_freq
# (no frequency scaling, so 4_thermal_si); the baseline figure keeps baseline3.
STAGES = [
    ("0_baseline",   BASE3,       "0_baseline",   "3D-stacking baseline (no optimizations)", "layerwise_baseline"),
    ("5_thermal_si", BASE_NOFREQ, "4_thermal_si", "after thermal-silicon optimization",      "layerwise_thermal_si"),
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


def grid(stack_dir, stage, layer):
    g = np.loadtxt(os.path.join(stack_dir, stage, f"{PREFIX}.layer{layer}")) - 273.15
    return g.reshape(ROWS, COLS)


def profile(stack_dir, stage):
    """[(tier, tech, peak_C), ...] for the memory dies, bottom (nearest GPU) -> top.
    Layer indices are read from THIS stage's LCF (base-die presence shifts them)."""
    lcf = os.path.join(stack_dir, stage, "hybrid_lcf.csv")
    mem = []
    with open(lcf) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "nv_tier_flp.csv":
                mem.append((int(idx), "Fe-RAM"))
            elif flp == "dram_tier_flp.csv":
                mem.append((int(idx), "DRAM"))
    mem.sort()
    return [(pos + 1, tech, float(grid(stack_dir, stage, lyr)[MASK].max()))
            for pos, (lyr, tech) in enumerate(mem)]


def make(hyb_stage, base_dir, base_stage, stage_title, out_name):
    dram = profile(base_dir, base_stage)   # pure DRAM
    hyb = profile(HERE, hyb_stage)         # φ-HBM hybrid
    n = len(hyb)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    # Fe-RAM region shading (hybrid bottom dies)
    n_nv = sum(1 for _, t, _ in hyb if t == "Fe-RAM")
    if n_nv:
        ax.axvspan(0.5, n_nv + 0.5, color=NV_RED, alpha=0.08, zorder=0)
        ax.text((n_nv + 1) / 2.0, 0.035, "Fe-RAM (bottom)", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=13, color=NV_RED, fontweight="bold")

    # pure DRAM stack — blue dashed line, square markers
    xd = [p for p, _, _ in dram]
    yd = [t for _, _, t in dram]
    ax.plot(xd, yd, "--", color=BLUE, lw=2.2, marker="s", ms=8, mec="white", mew=1.0, zorder=5)

    # hybrid stack — purple line; Fe-RAM tiers = red circles, DRAM tiers = orange squares
    xh = [p for p, _, _ in hyb]
    yh = [t for _, _, t in hyb]
    ax.plot(xh, yh, "-", color=PURPLE, lw=2.4, zorder=4)
    for p, tech, t in hyb:
        if tech == "Fe-RAM":
            ax.plot(p, t, "o", ms=9, color=NV_RED, mec="white", mew=1.2, zorder=6)
        else:
            ax.plot(p, t, "s", ms=8, color=DR_ORG, mec="white", mew=1.0, zorder=6)

    # tier-1 (bottom, hottest die) peak callouts, coloured to match each stack
    ax.annotate(f"{dram[0][2]:.1f} °C", (1, dram[0][2]), xytext=(8, 13), textcoords="offset points",
                ha="left", va="bottom", fontsize=15, fontweight="bold", color=BLUE, zorder=7)
    ax.annotate(f"{hyb[0][2]:.1f} °C", (1, hyb[0][2]), xytext=(8, 13), textcoords="offset points",
                ha="left", va="bottom", fontsize=15, fontweight="bold", color=PURPLE, zorder=7)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([str(i) for i in range(1, n + 1)], fontsize=12.5, color=INK,
                       fontweight="bold")
    ax.set_xlabel("Memory tier index   (1 = bottom, nearest GPU  →  top = lid)",
                  fontsize=14, color=INK, fontweight="bold")
    ax.set_ylabel("Peak die temperature (°C)", fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(0.4, n + 0.6)
    lo = min(min(yd), min(yh))
    hi = max(max(yd), max(yh))
    ax.set_ylim(lo - 3, hi + 6)
    ax.grid(True, color=MUTE, alpha=0.25, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=12.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontweight("bold")

    handles = [
        Line2D([0], [0], color=PURPLE, lw=2.4, label="Fe-RAM + DRAM hybrid stack"),
        Line2D([0], [0], color=NV_RED, lw=0, marker="o", ms=9, mec="white", label="↳ Fe-RAM tier (bottom)"),
        Line2D([0], [0], color=DR_ORG, lw=0, marker="s", ms=8, mec="white", label="↳ DRAM tier (continues on top)"),
        Line2D([0], [0], color=BLUE, lw=2.2, ls="--", marker="s", ms=8, mec="white",
               label="Pure DRAM stack (HBM only)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
              prop={"size": 11.5, "weight": "bold"})

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{out_name}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


def main():
    for hyb_stage, base_dir, base_stage, title, out_name in STAGES:
        make(hyb_stage, base_dir, base_stage, title, out_name)


if __name__ == "__main__":
    main()
