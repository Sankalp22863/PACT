"""
φ-HBM three-layer tilted heatmaps — GPU / peak NVDRAM / peak DRAM
=================================================================
Three side-by-side tilted (3D-rotated) temperature planes for the φ-HBM STCO
endpoint (5_thermal_si: 4 NVDRAM + 8 DRAM, 0.8× freq = 368 W):

  * GPU FEOL layer          (compute-die heat source)
  * peak NVDRAM tier        (hottest NVDRAM die — bottom, nearest GPU)
  * peak DRAM tier          (hottest DRAM die)

Rendered as flat tilted colour planes (paper Icepak style) with the rainbow
"jet" colormap on a shared banded scale + one shared colorbar (Max/Min marked).

Output: phi_heatmaps.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "5_thermal_si")
PREFIX = "hybrid.grid.steady"
ROWS, COLS = 44, 60
GPU_X, GPU_Y = 0.030, 0.022                 # die length (x) / width (y), m
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005
CMAP = plt.get_cmap("jet")                  # paper's rainbow colormap
N_BANDS = 13                                # discrete bands (Icepak-style colorbar)


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


def grid(layer):
    return (np.loadtxt(os.path.join(RUN, f"{PREFIX}.layer{layer}")) - 273.15).reshape(ROWS, COLS)


def find_layers():
    gpu, nv, dr = None, [], []
    for line in open(os.path.join(RUN, "hybrid_lcf.csv")).read().splitlines()[1:]:
        idx, flp = line.split(",")[:2]
        idx = int(idx)
        if flp == "gpu_feol_flp.csv" and gpu is None:
            gpu = idx
        elif flp == "nv_tier_flp.csv":
            nv.append(idx)
        elif flp == "dram_tier_flp.csv":
            dr.append(idx)
    nv_peak = max(nv, key=lambda l: grid(l)[MASK].max())
    dr_peak = max(dr, key=lambda l: grid(l)[MASK].max())
    return gpu, nv_peak, dr_peak


def tilted_plane(ax, g, norm):
    xs = np.linspace(0, GPU_X * 1e3, COLS)
    ys = np.linspace(0, GPU_Y * 1e3, ROWS)
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=CMAP(norm(g)), shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.view_init(elev=48, azim=-60)
    ax.set_box_aspect((1, GPU_Y / GPU_X, 0.04), zoom=1.15)   # fill the axes, less whitespace
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)          # transparent bbox so overlapping panels interlock


def main():
    gpu_l, nv_l, dr_l = find_layers()
    # (title, layer, accent colour matching each layer's thermal identity)
    panels = [("GPU layer",        gpu_l, "#c0392b"),
              ("Peak NVDRAM tier", nv_l, "#e08214"),
              ("Peak DRAM tier",   dr_l, "#1e8f5a")]
    grids = [grid(l) for _, l, _ in panels]
    vmin = min(float(g.min()) for g in grids)
    vmax = max(float(g.max()) for g in grids)
    levels = np.linspace(vmin, vmax, N_BANDS + 1)
    norm = BoundaryNorm(levels, CMAP.N)

    fig = plt.figure(figsize=(14.0, 4.8))
    fig.patch.set_facecolor("white")
    # single overall title, centred above the three panels
    fig.text(0.445, 0.93, "φ-HBM", ha="center", va="center", fontsize=20,
             fontweight="bold", color="#111111")
    # Fanned tilted planes: monotonic downward stagger matching the tilt, spaced
    # so the parallelograms leave just a teeny gap between them (no overlap).
    AXW = 0.40                 # axes (plane) width
    STEP_X = 0.25              # x advance per panel (tiny gap between planes)
    STEP_Y = -0.085            # downward stagger per panel (fan direction)
    for i, (title, _, accent) in enumerate(panels):
        g = grids[i]
        x0 = 0.0 + i * STEP_X
        y0 = 0.02 + i * STEP_Y
        cx = x0 + AXW / 2.0
        # heading + peak pill track each plane's position
        fig.text(cx, y0 + 0.86, title, ha="center", va="center", fontsize=14,
                 fontweight="bold", color="#1a1a1a")
        fig.text(cx, y0 + 0.795, f"Peak {g.max():.1f} °C", ha="center", va="center",
                 fontsize=12, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.32", fc=accent, ec="none"))
        # tilted heatmap plane (transparent bbox so overlaps interlock)
        ax = fig.add_axes([x0, y0, AXW, 0.78], projection="3d")
        ax.set_zorder(len(panels) - i)   # left (upper) panels draw in front, like fanned cards
        tilted_plane(ax, g, norm)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    cax = fig.add_axes([0.885, 0.38, 0.015, 0.52])
    cb = fig.colorbar(sm, cax=cax, ticks=levels[::2], boundaries=levels, spacing="proportional")
    cb.set_label("Temperature (°C)", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=9.5)
    cax.annotate(f"Max {vmax:.1f}", (0.5, 1.0), xycoords="axes fraction",
                 xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9,
                 fontweight="bold", color="#1a1a1a")
    cax.annotate(f"Min {vmin:.1f}", (0.5, 0.0), xycoords="axes fraction",
                 xytext=(0, -13), textcoords="offset points", ha="center", fontsize=9,
                 fontweight="bold", color="#1a1a1a")

    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"phi_heatmaps.{ext}")
        fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.08)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
