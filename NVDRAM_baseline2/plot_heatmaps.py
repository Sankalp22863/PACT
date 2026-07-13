"""
Layer heatmaps after thermal-silicon optimization  (NVDRAM_baseline2)
=====================================================================
Steady-state temperature maps of the GPU compute die and the bottom-most DRAM
tier, taken from the final (post-thermal-silicon) stage of the all-DRAM 30x22 mm
waterfall. This stack is ALL-DRAM, so there is no NVDRAM layer to map (the third
panel notes this).

Output: heatmaps_thermal_si.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "5_thermal_si")          # post-thermal-silicon
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y, MEM = 30.0, 22.0, 10.0                 # mm
ROWS, COLS = 44, 60

# stack footprints in mm (paper layout: 2 edge columns, 2 stacks each)
def stack_rects():
    EDGE, VOID = 1.0, 5.0                     # mm (VOID matches run_waterfall VOID_UM)
    cw = (GPU_X - 2 * EDGE - VOID) / 2.0      # stack-column width
    xl, xr = EDGE, EDGE + cw + VOID           # left/right column x-origins
    ymid = GPU_Y / 2.0
    return [(xl, 0, cw, ymid), (xl, ymid, cw, ymid),
            (xr, 0, cw, ymid), (xr, ymid, cw, ymid)]


def load_layer(idx):
    g = (np.loadtxt(os.path.join(STAGE, f"{PREFIX}.layer{idx}")) - 273.15).reshape(ROWS, COLS)
    return g


def draw(ax, g, title, cmap="inferno"):
    im = ax.imshow(g, origin="lower", extent=[0, GPU_X, 0, GPU_Y], cmap=cmap, aspect="equal")
    for (x, y, w, h) in stack_rects():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="cyan", lw=1.0, ls="--"))
    ax.set_title(f"{title}\npeak {g.max():.1f} °C   (min {g.min():.1f})", fontsize=11)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046 * GPU_Y / GPU_X, pad=0.03)
    cb.set_label("°C", fontsize=9)


def main():
    bd = int(open(os.path.join(HERE, "bottom_dram_layer.txt")).read().strip())
    gf = int(open(os.path.join(HERE, "gpu_feol_layer.txt")).read().strip())
    gpu = load_layer(gf)
    dram = load_layer(bd)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    draw(axes[0], gpu,  f"GPU compute die (layer {gf})")
    draw(axes[1], dram, f"Bottom-most DRAM tier (layer {bd})")

    # third panel: no NVDRAM layer in an all-DRAM stack
    axes[2].axis("off")
    axes[2].text(0.5, 0.5, "NVDRAM layer\n\n(not present —\nthis is an ALL-DRAM stack)",
                 ha="center", va="center", fontsize=13, color="#8a8a8a",
                 bbox=dict(boxstyle="round,pad=0.8", fc="#f4f4f4", ec="#c0c0c0"))

    fig.suptitle("Steady-state layer temperatures after thermal-silicon optimization "
                 "(all-DRAM, 30 × 22 mm; dashed = memory-stack footprints)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"heatmaps_thermal_si.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
