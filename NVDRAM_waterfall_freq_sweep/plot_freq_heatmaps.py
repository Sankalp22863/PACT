"""
GPU|DRAM temperature heatmaps for the GPU-frequency-scaling endpoints
====================================================================
Flat-plane jet surfaces reduced to the two panels the frequency-scaling story
tracks:

    GPU compute die (L2)   |   Lowest DRAM tier (L20)

One figure per GPU frequency endpoint of the 4 NVDRAM + 8 DRAM full-STCO stack
(the same physical stack; only GPU frequency -> GPU power changes).

Colour scheme (matches the reference φ-HBM figure exactly):
  * bright standard "jet" colormap, DISCRETISED into 13 bands
    (levels = linspace(vmin, vmax, 14)), so both the planes and the colorbar
    read as contour bands rather than a smooth gradient;
  * ONE shared vertical colorbar per figure, spanning that figure's global
    temperature range, labelled "Max <hi>" / "Min <lo>" at the extremes and
    "Temperature (°C)" on the side;
  * a coloured peak callout above each plane — red for the GPU layer, green for
    the DRAM tier (white bold "Peak <T> °C").

These two figures drop into the FROST slide boxes
"Heatmap for 1x / 0.8x scaling — GPU Temp | DRAM Temp".

Output (next to this script): heatmaps_1x_scaling.png/.pdf, heatmaps_0p8x_scaling.png/.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 30.0, 22.0                       # mm
ROWS, COLS = 44, 60
GPU_L, DRAM_L = 2, 20                            # GPU-FEOL, lowest DRAM tier (verified vs freq CSV)

ELEV, AZIM = 35, -60
BOX = (GPU_X, GPU_Y, GPU_X * 0.05)
NLEV = 13                                        # discrete jet bands (14 level boundaries)
CMAP = plt.get_cmap("jet", NLEV)                 # bright standard jet, discretised
INK = "#2b2b2b"
CALLOUT = {"gpu": "#c0392b", "dram": "#27ae60"}  # red GPU / green DRAM peak boxes

# freq -> (folder, GPU power W, output basename)
CONFIGS = [
    ("1.0", "NVDRAM_waterfall_freq_1p0", 414, "heatmaps_1x_scaling"),
    ("0.8", "NVDRAM_waterfall_freq_0p8", 368, "heatmaps_0p8x_scaling"),
]


def load(folder, idx):
    p = os.path.join(HERE, folder, f"{PREFIX}.layer{idx}")
    return (np.loadtxt(p) - 273.15).reshape(ROWS, COLS)


def panel(ax, g, title, callout_color, norm):
    xs = np.linspace(0, GPU_X, COLS)
    ys = np.linspace(0, GPU_Y, ROWS)
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=CMAP(norm(g)), shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_zticks([])
    ax.set_box_aspect(BOX)
    ax.set_title(title, fontsize=13, color=INK, pad=0)
    # coloured peak callout above the plane
    ax.text2D(0.5, 0.84, f"Peak {g.max():.1f} °C", transform=ax.transAxes,
              ha="center", va="top", fontsize=12, fontweight="bold", color="white",
              bbox=dict(boxstyle="round,pad=0.35", fc=callout_color, ec=callout_color))


def make(freq, folder, power, base):
    gpu = load(folder, GPU_L)
    dram = load(folder, DRAM_L)
    vmin = float(min(gpu.min(), dram.min()))
    vmax = float(max(gpu.max(), dram.max()))
    levels = np.linspace(vmin, vmax, NLEV + 1)
    norm = BoundaryNorm(levels, ncolors=NLEV)

    fig = plt.figure(figsize=(13.0, 5.9))
    fig.suptitle(
        f"GPU frequency {freq}× ({power} W) — "
        f"φ-HBM 4 NVDRAM + 8 DRAM (full STCO), 30 × 22 mm",
        fontsize=13.5, color=INK, y=0.98)

    ax1 = fig.add_axes([0.015, 0.04, 0.42, 0.80], projection="3d")
    panel(ax1, gpu, "GPU layer", CALLOUT["gpu"], norm)
    ax2 = fig.add_axes([0.45, 0.04, 0.42, 0.80], projection="3d")
    panel(ax2, dram, "Lowest DRAM tier", CALLOUT["dram"], norm)

    # single shared discrete colorbar
    cax = fig.add_axes([0.905, 0.17, 0.018, 0.60])
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, boundaries=levels, ticks=levels[::2])
    cbar.set_ticklabels([f"{t:.2f}" for t in levels[::2]])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Temperature (°C)", fontsize=11, color=INK)
    cbar.outline.set_linewidth(0.6)
    cax.text(0.5, 1.04, f"Max {vmax:.1f}", transform=cax.transAxes,
             ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    cax.text(0.5, -0.05, f"Min {vmin:.1f}", transform=cax.transAxes,
             ha="center", va="top", fontsize=10, fontweight="bold", color=INK)

    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{base}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def main():
    for freq, folder, power, base in CONFIGS:
        make(freq, folder, power, base)


if __name__ == "__main__":
    main()
