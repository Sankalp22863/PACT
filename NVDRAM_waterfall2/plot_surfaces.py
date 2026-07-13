"""
Paper-style layer temperature heatmaps after thermal-silicon optimization
=========================================================================
Reproduces the reference IEDM heatmap style (see PACT/Experiments/scripts/
pact_plots.py :: surface_plane / _flat_plane): each layer is drawn as a tilted
FLAT colour plane (colour = temperature, z = 0), jet colormap (blue = coolest →
red = hottest), a near-flat 3D box, a red "Peak Temperature" callout, and a
per-panel "Temperature (°C)" colorbar.

Auto-adapts to the stack: one panel per available memory tier — GPU compute die,
lowest NVDRAM tier (if present), lowest DRAM tier — read from the *_layer.txt
files written by run_waterfall.py. Uses only the final (thermal-silicon) stage.
Output: surfaces_thermal_si.png / .pdf

This is the canonical heatmap style; future experiments should copy this script.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "5_thermal_si")     # final optimization only
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 30.0, 22.0                       # mm
ROWS, COLS = 44, 60

# ---- reference (pact_plots.py) style constants ----
ELEV, AZIM = 35, -60                            # reference view_init
BOX = (GPU_X, GPU_Y, GPU_X * 0.05)              # reference (1,1,0.05) flatness, scaled to the rectangle
PEAK_C = "#b30000"                              # reference peak-callout red

# jet, deepened so the colours read richer (not washed out); used for BOTH the
# plane facecolors and the colorbar so they stay consistent.
DARKEN = 0.72                                   # multiply colormap RGB to deepen
_jc = plt.get_cmap("jet")(np.linspace(0, 1, 256))
_jc[:, :3] *= DARKEN
CMAP = ListedColormap(_jc)                      # blue (coolest) -> red (hottest), deepened


def load(idx):
    return (np.loadtxt(os.path.join(STAGE, f"{PREFIX}.layer{idx}")) - 273.15).reshape(ROWS, COLS)


def read_layer(fname):
    p = os.path.join(HERE, fname)
    return int(open(p).read().strip()) if os.path.exists(p) else None


def panel(fig, ax, g, title):
    xs = np.linspace(0, GPU_X, COLS)
    ys = np.linspace(0, GPU_Y, ROWS)
    X, Y = np.meshgrid(xs, ys)
    cmap = CMAP
    norm = Normalize(vmin=g.min(), vmax=g.max())
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=cmap(norm(g)), shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=True)   # smooth, no mesh
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_zticks([])
    ax.set_box_aspect(BOX)
    ax.set_title(title, fontsize=12, pad=2)
    ax.text2D(0.5, -0.03, f"Peak Temperature\n{g.max():.1f} °C", transform=ax.transAxes,
              ha="center", va="top", fontsize=11.5, fontweight="bold", color=PEAK_C,
              bbox=dict(boxstyle="round", fc="white", ec=PEAK_C, alpha=0.9))
    ax.text2D(0.5, 0.02, f"← {GPU_X:g} × {GPU_Y:g} mm →", transform=ax.transAxes,
              ha="center", va="bottom", fontsize=8, color="#444444")
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(g)
    fig.colorbar(m, ax=ax, label="Temperature (°C)", fraction=0.03, pad=0.10, shrink=0.6)


def main():
    gf = read_layer("gpu_feol_layer.txt")
    nv = read_layer("nv_tier_layer.txt")
    bd = read_layer("bottom_dram_layer.txt")

    panels = [(gf, "GPU compute die")]
    if nv is not None:
        panels.append((nv, "Lowest NVDRAM tier"))
    panels.append((bd, "Lowest DRAM tier"))

    n = len(panels)
    fig = plt.figure(figsize=(6.4 * n, 6.0))
    for pos, (lid, label) in enumerate(panels, 1):
        ax = fig.add_subplot(1, n, pos, projection="3d")
        panel(fig, ax, load(lid), f"{label} — L{lid}")

    kind = ("φ-HBM: 4 NVDRAM + 8 DRAM" if nv is not None else "all-DRAM")
    fig.suptitle(f"Steady-state layer temperatures after thermal-silicon optimization "
                 f"({kind}, {GPU_X:g} × {GPU_Y:g} mm)", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"surfaces_thermal_si.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
