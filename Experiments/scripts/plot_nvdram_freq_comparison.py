"""
4 NV + 8 DRAM — effect of GPU frequency scaling (vs pure-DRAM baseline)
======================================================================
Layerwise per-die peak-temperature profile up the 3D stack, comparing the
4 NVDRAM + 8 DRAM composition WITH and WITHOUT GPU frequency scaling, against the
pure-DRAM baseline:

  * 0 NV + 12 DRAM (pure-DRAM baseline)  = ../../NVDRAM_baseline_no_freq/4_thermal_si
      414 W, no frequency scaling (STCO endpoint)          -> tier-1 peak 100.2 C
  * 4 NV + 8 DRAM  (414 W, no freq)      = ../EXP_NVDRAM_4_STCO
      STCO endpoint, GPU at full 414 W                     -> tier-1 peak  94.7 C
  * 4 NV + 8 DRAM  (phi-HBM, 0.8x freq)  = ../../NVDRAM_waterfall_freq_sweep/
                                             NVDRAM_waterfall_freq_0p8
      STCO endpoint with 0.8x GPU frequency scaling        -> tier-1 peak  88.1 C

Die-type markers: NVDRAM tier = circle, DRAM tier = square. Each die temperature
is the peak under the stack columns (same under-stack mask as the waterfall).
Output: ../nvdram_4NV_freq_comparison.png / .pdf

Figure geometry
---------------
The paper places this at width=\\linewidth in a 0.60\\textwidth minipage
(= 4.285 in) with height=2.53 in. Aspect must stay >= 4.285/2.53 = 1.694 so the
graphic is WIDTH-bound and fills the column; below that it goes height-bound and
leaves dead side gutters (the old 1.536 wasted 0.40 in). Keep figsize wide.

Readability rules (do not regress)
----------------------------------
* No annotation text may sit on a same-coloured curve -- that was the original
  defect: purple "0.8x frequency scaling" lay directly on the purple line.
* Curve colours stay as-is (thick strokes read fine), but TEXT uses darkened
  variants: raw #d4a017 gold on white is ~2.6:1 contrast, well under WCAG 4.5:1.
* Every floating label carries a translucent white bbox so it never fights a
  gridline, threshold or curve underneath.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))          # Experiments/scripts
EXP = os.path.abspath(os.path.join(HERE, ".."))            # Experiments
PACT = os.path.abspath(os.path.join(EXP, ".."))            # PACT
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005    # under-stack mask geometry (matches run_waterfall)

INK, MUTE = "#2b2b2b", "#8a8a8a"
RED, ORANGE = "#c0392b", "#e08214"

# (label, grid-dir, curve colour, text colour) — bottom(hottest) -> top
COMPS = [
    ("0 NV + 12 DRAM (pure-DRAM baseline)",
     os.path.join(PACT, "NVDRAM_baseline_no_freq", "4_thermal_si"), "#1f77b4", "#14537d"),
    ("4 NV + 8 DRAM  — 414 W, no freq scaling",
     os.path.join(EXP, "EXP_NVDRAM_4_STCO"), "#d4a017", "#8a6a0a"),
    ("4 NV + 8 DRAM  — φ-HBM, 0.8× freq scaling",
     os.path.join(PACT, "NVDRAM_waterfall_freq_sweep", "NVDRAM_waterfall_freq_0p8"),
     "#7d3c98", "#5b2a70"),
]

BOX = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82)
# For labels that unavoidably cross a line (the tier-1 callouts sit where the
# ellipse pinches to +/-1.6 C, too narrow to fit inside): a white glyph halo
# keeps them legible without punching a rectangular hole in the dashes.
HALO = [pe.withStroke(linewidth=3.4, foreground="white")]


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


def profile(grid_dir):
    """[(tier, tech, peak_C), ...] for the memory dies, bottom (nearest GPU) -> top."""
    lcf = os.path.join(grid_dir, "hybrid_lcf.csv")
    mem = []
    with open(lcf) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "nv_tier_flp.csv":
                mem.append((int(idx), "NVDRAM"))
            elif flp == "dram_tier_flp.csv":
                mem.append((int(idx), "DRAM"))
    mem.sort()
    out = []
    for pos, (lyr, tech) in enumerate(mem):
        g = (np.loadtxt(os.path.join(grid_dir, f"{PREFIX}.layer{lyr}")) - 273.15).reshape(ROWS, COLS)
        out.append((pos + 1, tech, float(g[MASK].max())))
    return out


def visual_ellipse(fig, ax, p0, p1, pad_in=0.30, minor_in=0.74, **kw):
    """Dashed ellipse whose major axis hugs the segment p0->p1 (both in data coords).

    Built in DISPLAY space and mapped back, because the axes are not equal-aspect:
    a matplotlib Ellipse with an angle in data units comes out sheared. Call only
    after the limits are final and the figure has been drawn once.
    """
    (x0, y0), (x1, y1) = ax.transData.transform(p0), ax.transData.transform(p1)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ang = np.arctan2(y1 - y0, x1 - x0)
    a = np.hypot(x1 - x0, y1 - y0) / 2.0 + pad_in * fig.dpi
    b = minor_in * fig.dpi / 2.0
    t = np.linspace(0, 2 * np.pi, 361)
    ex, ey = a * np.cos(t), b * np.sin(t)
    px = cx + ex * np.cos(ang) - ey * np.sin(ang)
    py = cy + ex * np.sin(ang) + ey * np.cos(ang)
    back = ax.transData.inverted().transform(np.column_stack([px, py]))
    ax.plot(back[:, 0], back[:, 1], **kw)


def main():
    data = [(label, cc, tc, profile(d)) for (label, d, cc, tc) in COMPS]

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(60, 106)

    # ---- thermal thresholds (under everything) -------------------------------
    for yv, col, txt in ((90, RED, "90 °C"), (85, ORANGE, "85 °C")):
        ax.axhline(yv, color=col, ls=":", lw=1.8, zorder=2)
        ax.text(12.44, yv + 0.5, txt, ha="right", va="bottom", fontsize=9.5,
                fontweight="bold", color=col, zorder=9, bbox=BOX)

    # ---- curves + die-type markers ------------------------------------------
    for label, cc, tc, dies in data:
        ax.plot([p for p, _, _ in dies], [t for _, _, t in dies], "-", color=cc, lw=2.0, zorder=4)
        for p, tech, t in dies:
            ax.plot(p, t, "o" if tech == "NVDRAM" else "s", ms=8, color=cc,
                    mec="white", mew=1.0, zorder=6)

    # ---- tier-1 (hottest die) peak callouts ---------------------------------
    # UP-right, never down: at tier 1 every curve is descending steeply, so a
    # label below-right lands on its own line and the white bbox then eats it.
    # Above each point sits a 5.5-6.6 °C inter-curve band -- ample. The blue one
    # lands inside the ellipse's white interior, clear of its dashed outline.
    for label, cc, tc, dies in data:
        t1 = dies[0][2]
        ax.annotate(f"{t1:.1f}°C", (1, t1), xytext=(9, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10.5, fontweight="bold",
                    color=tc, zorder=10, path_effects=HALO)

    # ---- axes ---------------------------------------------------------------
    n_tiers = max(len(d[3]) for d in data)
    ax.set_xticks(range(1, n_tiers + 1))
    ax.set_xticklabels([str(i) for i in range(1, n_tiers + 1)], fontsize=9.5, color=INK)
    ax.set_xlabel("Memory die position   (Base = nearest GPU  →  top tier = nearest cooling lid)",
                  fontsize=11, color=INK)
    ax.set_ylabel("Peak die temperature (°C)", fontsize=12, color=INK)
    ax.grid(True, color=MUTE, alpha=0.3, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    # ---- legends ------------------------------------------------------------
    leg1 = ax.legend(handles=[Line2D([0], [0], color=cc, lw=2.4, label=lb) for lb, cc, _, _ in data],
                     title="Configuration", loc="upper right", frameon=True,
                     fontsize=9.0, title_fontsize=9.5, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=[
        Line2D([0], [0], color=MUTE, lw=0, marker="o", ms=9, mfc="#bbb", mec="white", label="NVDRAM tier"),
        Line2D([0], [0], color=MUTE, lw=0, marker="s", ms=8, mfc="#bbb", mec="white", label="DRAM tier"),
    ], title="Die type", loc="lower left", frameon=True, fontsize=9.5,
        title_fontsize=10, framealpha=0.95)

    # ---- "0.8x frequency scaling": gold -> purple drop at tier 3 ------------
    # Label goes LEFT of the arrow (ha=right): the wedge left of x=2.5 below the
    # purple curve is the only pocket here that no curve crosses.
    xf = 2.6
    gold_y = np.interp(xf, [p for p, _, _ in data[1][3]], [t for _, _, t in data[1][3]])
    purp_y = np.interp(xf, [p for p, _, _ in data[2][3]], [t for _, _, t in data[2][3]])
    ax.annotate("", xy=(xf, purp_y + 0.5), xytext=(xf, gold_y - 0.5),
                arrowprops=dict(arrowstyle="-|>", color=data[2][2], lw=2.0,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text(xf - 0.15, 82.6, "0.8× frequency\nscaling", ha="right", va="center",
            fontsize=9.5, fontweight="bold", color=data[2][2], zorder=10, bbox=BOX,
            linespacing=1.25)

    # ---- "thermal headroom": blue -> gold gap where the baseline hits 90 ----
    xh = 5.0
    b_y = np.interp(xh, [p for p, _, _ in data[0][3]], [t for _, _, t in data[0][3]])
    g_y = np.interp(xh, [p for p, _, _ in data[1][3]], [t for _, _, t in data[1][3]])
    ax.annotate("", xy=(xh, b_y), xytext=(xh, g_y),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=7)
    # Text goes up-right into the wedge above the baseline, NOT beside the arrow:
    # the blue curve runs straight through the blue/gold band here.
    ax.text(5.5, 93.0, "thermal\nheadroom", ha="left", va="center",
            fontsize=9, fontstyle="italic", fontweight="bold", color=INK,
            zorder=10, bbox=BOX, linespacing=1.25)

    # ---- "thermally constrained" ellipse over the baseline's T>90 run -------
    fig.tight_layout()
    fig.canvas.draw()                      # transData must be final before mapping
    visual_ellipse(fig, ax, (1, 100.23), (5, 90.2), pad_in=0.30, minor_in=0.78,
                   ls=(0, (7, 5)), color=INK, lw=1.7, zorder=5)
    # Sits ABOVE the ellipse: the old spot (upper right) collided with the
    # legend, and the wedge right of the ellipse is now the headroom label's.
    ax.annotate("Thermally constrained\n(T > 90 °C)", xy=(2.60, 100.0),
                xytext=(2.15, 103.6), ha="left", va="center", fontsize=9.5,
                fontweight="bold", color=RED, zorder=10, bbox=BOX, linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.1,
                                connectionstyle="arc3,rad=-0.2", shrinkB=1))

    for ext in ("png", "pdf"):
        out = os.path.join(EXP, f"nvdram_4NV_freq_comparison.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
