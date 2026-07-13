"""
3D-perspective temperature heatmaps after thermal-silicon optimization  (NVDRAM_waterfall2)
===========================================================================================
Same flat top-down heatmaps as plot_heatmaps.py (inferno colormap, stack-footprint
boxes), rendered as a tilted flat colored plane for a 3D look (temperature = color,
NOT height). GPU compute die + lowest NVDRAM tier + lowest DRAM tier, post-thermal-
silicon. Output: surfaces_thermal_si.png / .pdf
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
    for (x0, y0, w, h) in stack_rects():                 # stack-footprint outlines on the plane
        ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + h, y0 + h, y0], [0] * 5,
                color="black", lw=1.1, ls="--", zorder=5)
    ax.view_init(elev=42, azim=-62)
    ax.set_zlim(-1, 1)
    ax.set_zticks([])
    ax.set_box_aspect((GPU_X, GPU_Y, 6))         # flatten the 3D box so the plane fills the panel
    ax.set_xlabel("x (mm)", fontsize=9, labelpad=1)
    ax.set_ylabel("y (mm)", fontsize=9, labelpad=1)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\npeak {g.max():.1f} °C", fontsize=11, pad=0)
    mp = cm.ScalarMappable(norm=norm, cmap=CMAP); mp.set_array(g)
    fig.colorbar(mp, ax=ax, shrink=0.5, pad=0.06, label="Temperature (°C)")


def main():
    gf = int(open(os.path.join(HERE, "gpu_feol_layer.txt")).read())
    nv = int(open(os.path.join(HERE, "nv_tier_layer.txt")).read())
    bd = int(open(os.path.join(HERE, "bottom_dram_layer.txt")).read())

    fig = plt.figure(figsize=(16.5, 4.8))
    panels = [(load(gf), f"GPU compute die (layer {gf})"),
              (load(nv), f"Lowest NVDRAM tier (layer {nv})"),
              (load(bd), f"Lowest DRAM tier (layer {bd})")]
    for pos, (g, t) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, pos, projection="3d")
        plane(fig, ax, g, t)
    fig.suptitle("Steady-state layer temperatures after thermal-silicon optimization "
                 "(φ-HBM: 4 NVDRAM + 8 DRAM, 30 × 22 mm; dashed = memory-stack footprints)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"surfaces_thermal_si.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
