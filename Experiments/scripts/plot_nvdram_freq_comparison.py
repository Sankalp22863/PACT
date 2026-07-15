"""
4 NV + 8 DRAM — effect of GPU frequency scaling (vs pure-DRAM baseline)
======================================================================
Layerwise per-die peak-temperature profile up the 3D stack, comparing the
4 NVDRAM + 8 DRAM composition WITH and WITHOUT GPU frequency scaling, against the
pure-DRAM baseline:

  * 0 NV + 12 DRAM (pure-DRAM baseline)  = ../../NVDRAM_baseline_no_freq/4_thermal_si
      414 W, no frequency scaling (STCO endpoint)
  * 4 NV + 8 DRAM  (414 W, no freq)      = ../EXP_NVDRAM_4_STCO
      STCO endpoint, GPU at full 414 W
  * 4 NV + 8 DRAM  (phi-HBM, 0.8x freq)  = ../../NVDRAM_waterfall3/5_thermal_si
      STCO endpoint with 0.8x GPU frequency scaling (414 W -> 368 W, paper Fig. 8)

Die-type markers: NVDRAM tier = circle, DRAM tier = square. Each die temperature
is the peak under the stack columns (same under-stack mask as the waterfall).
Output: ../nvdram_4NV_freq_comparison.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))          # Experiments/scripts
EXP = os.path.abspath(os.path.join(HERE, ".."))            # Experiments
PACT = os.path.abspath(os.path.join(EXP, ".."))            # PACT
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005    # under-stack mask geometry (matches run_waterfall)

INK, MUTE = "#2b2b2b", "#8a8a8a"

# (label, grid-dir, colour) — bottom(hottest) -> top
COMPS = [
    ("0 NV + 12 DRAM (pure-DRAM baseline)",
     os.path.join(PACT, "NVDRAM_baseline_no_freq", "4_thermal_si"), "#1f77b4"),
    ("4 NV + 8 DRAM  — 414 W, no freq scaling",
     os.path.join(EXP, "EXP_NVDRAM_4_STCO"), "#d4a017"),
    ("4 NV + 8 DRAM  — φ-HBM, 0.8× freq scaling",
     os.path.join(PACT, "NVDRAM_waterfall3", "5_thermal_si"), "#7d3c98"),
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


def main():
    data = [(label, color, profile(d)) for (label, d, color) in COMPS]

    fig, ax = plt.subplots(figsize=(11.0, 7.2), dpi=200)

    for label, color, dies in data:
        xs = [p for p, _, _ in dies]
        ys = [t for _, _, t in dies]
        ax.plot(xs, ys, "-", color=color, lw=2.0, zorder=4)
        for p, tech, t in dies:
            mk = "o" if tech == "NVDRAM" else "s"
            ax.plot(p, t, mk, ms=8, color=color, mec="white", mew=1.0, zorder=6)

    # tier-1 (bottom, hottest die) peak callouts, colour-matched
    for label, color, dies in data:
        t1 = dies[0][2]
        ax.annotate(f"{t1:.1f}°C", (1, t1), xytext=(8, 8), textcoords="offset points",
                    ha="left", va="bottom", fontsize=15, fontweight="bold", color=color, zorder=8)

    # DRAM thermal-limit cutoffs
    n_tiers = max(len(d[2]) for d in data)
    for temp_c, col in ((90.0, "#b30000"), (85.0, "#e08214")):
        ax.axhline(temp_c, ls=":", lw=2.0, color=col, alpha=0.9, zorder=2)
        ax.annotate(f"{temp_c:.0f} °C", (n_tiers + 0.42, temp_c), ha="right", va="bottom",
                    fontsize=12, fontweight="bold", color=col, zorder=8)

    ax.set_xticks(range(1, n_tiers + 1))
    ax.set_xticklabels([str(i) for i in range(1, n_tiers + 1)], fontsize=12.5, color=INK,
                       fontweight="bold")
    ax.set_xlabel("Memory die position   (Base = nearest GPU  →  top tier = nearest cooling lid)",
                  fontsize=14, color=INK, fontweight="bold")
    ax.set_ylabel("Peak die temperature (°C)", fontsize=14, color=INK, fontweight="bold")
    ax.set_xlim(0.5, n_tiers + 0.5)
    all_t = [t for _, _, dies in data for _, _, t in dies]
    ax.set_ylim(min(all_t) - 4, max(all_t) + 6)
    ax.grid(True, color=MUTE, alpha=0.3, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelsize=12.5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontweight("bold")

    # legend 1 — configuration (colours)
    comp_handles = [Line2D([0], [0], color=color, lw=2.4, label=label) for label, color, _ in data]
    leg1 = ax.legend(handles=comp_handles, title="Configuration", loc="upper right",
                     frameon=True, framealpha=0.95,
                     prop={"size": 11.5, "weight": "bold"}, title_fontsize=12)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)

    # legend 2 — die type (marker shapes)
    type_handles = [
        Line2D([0], [0], color=MUTE, lw=0, marker="o", ms=9, mfc="#bbb", mec="white", label="NVDRAM tier"),
        Line2D([0], [0], color=MUTE, lw=0, marker="s", ms=8, mfc="#bbb", mec="white", label="DRAM tier"),
    ]
    leg2 = ax.legend(handles=type_handles, title="Die type", loc="lower left",
                     frameon=True, framealpha=0.95,
                     prop={"size": 11.5, "weight": "bold"}, title_fontsize=12)
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout()

    # thermally-constrained oval around the 4 hottest pure-DRAM dies (tiers 1-4),
    # built in display space so it hugs the sloped curve under unequal axis scales
    import math
    from matplotlib.transforms import IdentityTransform
    fig.canvas.draw()
    pure = data[0][2]
    pts = ax.transData.transform([(p, t) for p, _, t in pure[:4]])
    cx, cy = pts.mean(axis=0)
    dx, dy = pts[-1] - pts[0]
    ell = Ellipse((cx, cy), width=math.hypot(dx, dy) + 170, height=115,
                  angle=math.degrees(math.atan2(dy, dx)), fill=False,
                  ec="#444444", lw=2.4, ls="--", zorder=7,
                  transform=IdentityTransform(), clip_on=False)
    ax.add_patch(ell)
    ax.annotate("Thermally constrained\n(T > 90 °C)", (3.15, 97.2),
                xytext=(4.35, 101.8), ha="left", va="center",
                fontsize=13, fontweight="bold", color="#b30000", zorder=8,
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.8))

    # thermal headroom (pure-DRAM vs hybrid @ 414 W) and the frequency-scaling
    # drop it buys (hybrid 414 W -> phi-HBM scaled)
    tiers = [pt for pt, _, _ in data[0][2]]
    y_blue = [t for _, _, t in data[0][2]]
    y_yel = [t for _, _, t in data[1][2]]
    y_pur = [t for _, _, t in data[2][2]]
    hx = 4.6
    h_hi, h_lo = np.interp(hx, tiers, y_blue), np.interp(hx, tiers, y_yel)
    ax.annotate("", (hx, h_hi - 0.25), xytext=(hx, h_lo + 0.25),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=2.2,
                                mutation_scale=16), zorder=7)
    ax.annotate("thermal\nheadroom", (5.0, 82.6), ha="left", va="center",
                fontsize=13, fontweight="bold", fontstyle="italic", color=INK, zorder=8)
    fx = 2.6
    f_hi, f_lo = np.interp(fx, tiers, y_yel), np.interp(fx, tiers, y_pur)
    ax.annotate("", (fx, f_lo + 0.35), xytext=(fx, f_hi - 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#7d3c98", lw=2.2,
                                mutation_scale=18), zorder=7)
    ax.annotate("0.8× frequency\nscaling", (2.78, 80.0), ha="left", va="center",
                fontsize=13, fontweight="bold", color="#7d3c98", zorder=8)

    for ext in ("png", "pdf"):
        out = os.path.join(EXP, f"nvdram_4NV_freq_comparison.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
