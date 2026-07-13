"""
3D-perspective temperature heatmaps after thermal-silicon optimization  (NVDRAM_baseline2)
==========================================================================================
All-DRAM stack: GPU compute die + lowest DRAM tier as flat top-down heatmaps
(jet colormap: blue = coolest, red = hottest; stack-footprint boxes) rendered as a
tilted flat colored plane for a 3D look. Third panel notes there is no NVDRAM layer.
Output: surfaces_thermal_si.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "5_thermal_si")
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 30.0, 22.0          # mm
ROWS, COLS = 44, 60
CMAP = "jet"          # blue (coolest) -> red (hottest)


def load(idx):
    return (np.loadtxt(os.path.join(STAGE, f"{PREFIX}.layer{idx}")) - 273.15).reshape(ROWS, COLS)


def stack_rects():
    EDGE, VOID = 1.0, 5.0
    cw = (GPU_X - 2 * EDGE - VOID) / 2.0
    xl, xr = EDGE, EDGE + cw + VOID
    ymid = GPU_Y / 2.0
    return [(xl, 0, cw, ymid), (xl, ymid, cw, ymid), (xr, 0, cw, ymid), (xr, ymid, cw, ymid)]


def plane(fig, ax, g, title):
    xs = np.linspace(0, GPU_X, COLS)
    ys = np.linspace(0, GPU_Y, ROWS)
    X, Y = np.meshgrid(xs, ys)
    norm = Normalize(vmin=g.min(), vmax=g.max())
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=plt.get_cmap(CMAP)(norm(g)),
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    for (x0, y0, w, h) in stack_rects():
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + h, y0 + h, y0], [0] * 5,
                color="black", lw=1.1, ls="--", zorder=5)
    ax.view_init(elev=42, azim=-62)
    ax.set_zlim(-1, 1)
    ax.set_zticks([])
    ax.set_box_aspect((GPU_X, GPU_Y, 6))
    ax.set_xlabel("x (mm)", fontsize=9, labelpad=1)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=1)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\npeak {g.max():.1f} °C", fontsize=11, pad=0)
    mp = cm.ScalarMappable(norm=norm, cmap=CMAP); mp.set_array(g)
    fig.colorbar(mp, ax=ax, shrink=0.5, pad=0.06, label="Temperature (°C)")


def main():
    gf = int(open(os.path.join(HERE, "gpu_feol_layer.txt")).read())
    bd = int(open(os.path.join(HERE, "bottom_dram_layer.txt")).read())

    fig = plt.figure(figsize=(16.5, 4.8))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plane(fig, ax1, load(gf), f"GPU compute die (layer {gf})")

    ax2 = fig.add_subplot(1, 3, 2)          # no NVDRAM layer in an all-DRAM stack
    ax2.axis("off")
    ax2.text(0.5, 0.5, "NVDRAM layer\n\n(not present —\nthis is an ALL-DRAM stack)",
             ha="center", va="center", fontsize=13, color="#8a8a8a",
             bbox=dict(boxstyle="round,pad=0.8", fc="#f4f4f4", ec="#c0c0c0"))

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plane(fig, ax3, load(bd), f"Lowest DRAM tier (layer {bd})")

    fig.suptitle("Steady-state layer temperatures after thermal-silicon optimization "
                 "(all-DRAM, 30 × 22 mm; dashed = memory-stack footprints)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"surfaces_thermal_si.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
