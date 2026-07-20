"""
4 NV + 8 DRAM — effect of GPU frequency scaling (vs pure-DRAM baseline)
======================================================================
Layerwise per-die peak-temperature profile up the 3D stack, comparing the
4 FeRAM + 8 DRAM composition WITH and WITHOUT GPU frequency scaling, against the
pure-DRAM baseline:

  * 0 NV + 12 DRAM (pure-DRAM baseline)  = ../../NVDRAM_baseline_no_freq/4_thermal_si
      414 W, no frequency scaling (STCO endpoint)
  * 4 NV + 8 DRAM  (414 W, no freq)      = ../EXP_NVDRAM_4_STCO
      STCO endpoint, GPU at full 414 W
  * 4 NV + 8 DRAM  (phi-HBM, 0.8x freq)  = ../../NVDRAM_waterfall3/5_thermal_si
      STCO endpoint with 0.8x GPU frequency scaling (414 W -> 368 W, paper Fig. 8)

Die-type markers: FeRAM (FeRAM) tier = circle, DRAM tier = square. Each die temperature
is the peak under the stack columns (same under-stack mask as the waterfall).
Output: ../nvdram_4NV_freq_comparison.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle

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
    ("0 FeRAM + 12 DRAM (pure-DRAM baseline)",
     os.path.join(PACT, "NVDRAM_baseline_no_freq", "4_thermal_si"), "#1f77b4"),
    ("4 FeRAM + 8 DRAM  — no freq scaling",
     os.path.join(EXP, "EXP_NVDRAM_4_STCO"), "#d4a017"),
    ("4 FeRAM + 8 DRAM  — XBM, 0.8× freq scaling",
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
                mem.append((int(idx), "FeRAM"))
            elif flp == "dram_tier_flp.csv":
                mem.append((int(idx), "DRAM"))
    mem.sort()
    out = []
    for pos, (lyr, tech) in enumerate(mem):
        g = (np.loadtxt(os.path.join(grid_dir, f"{PREFIX}.layer{lyr}")) - 273.15).reshape(ROWS, COLS)
        out.append((pos + 1, tech, float(g[MASK].max())))
    return out


# ── optional inset: top-down die layout (drawn from the model's floorplan) ────
LAYOUT_FLP = os.path.join(PACT, "NVDRAM_waterfall3", "5_thermal_si", "nv_tier_flp.csv")
LAY_GPU_X, LAY_GPU_Y = 0.030, 0.022        # die length (x) / width (y), m


def _read_flp(path):
    out = []
    for line in open(path).read().splitlines()[1:]:
        c = [t.strip() for t in line.split(",")]
        if len(c) >= 7:
            out.append((c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), c[6]))
    return out


def add_layout_inset(ax):
    """Top-down die layout inset: memory stack columns + central thermal-Si spine."""
    COL = {"MERGE_SI": "#2c6fa6", "THERMAL_SI": "#9ecae1"}
    iax = ax.inset_axes([0.015, 0.03, 0.265, 0.28])
    mm = 1e3
    for name, x, y, l, w, label in _read_flp(LAYOUT_FLP):
        fc = "#3cab6d" if name.startswith("MEM") else COL.get(label, "#dddddd")
        iax.add_patch(Rectangle((x * mm, y * mm), l * mm, w * mm, fc=fc,
                                ec="white", lw=0.8, zorder=2))
    iax.add_patch(Rectangle((0, 0), LAY_GPU_X * mm, LAY_GPU_Y * mm, fc="none",
                            ec=INK, lw=1.6, zorder=3))
    iax.text(LAY_GPU_X * mm * 0.5, LAY_GPU_Y * mm * 0.5, "Merged thermal Si",
             ha="center", va="center", rotation=90, fontsize=11,
             fontweight="bold", color="white", zorder=4)
    for cx in (LAY_GPU_X * mm * 0.20, LAY_GPU_X * mm * 0.80):
        iax.text(cx, LAY_GPU_Y * mm * 0.85, "Memory\nstack", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="white", zorder=4)
        iax.text(cx, LAY_GPU_Y * mm * 0.15, "4 FeRAM\n+ 8 DRAM", ha="center", va="center",
                 fontsize=11, fontweight="bold", color="white", zorder=4)
    iax.set_xlim(-0.5, LAY_GPU_X * mm + 0.5)
    iax.set_ylim(-0.5, LAY_GPU_Y * mm + 0.5)
    iax.set_aspect("equal")
    iax.axis("off")
    # caption sits to the RIGHT of the layout box, clear of the y-axis
    iax.text(1.04, 0.99, "Top-down\ndie layout\n(30 × 22 mm\nGPU)", transform=iax.transAxes,
             ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK,
             linespacing=1.25)


def main(with_layout=False):
    data = [(label, color, profile(d)) for (label, d, color) in COMPS]

    fig, ax = plt.subplots(figsize=(11.0, 7.2), dpi=200)

    for label, color, dies in data:
        xs = [p for p, _, _ in dies]
        ys = [t for _, _, t in dies]
        ax.plot(xs, ys, "-", color=color, lw=2.0, zorder=4)
        for p, tech, t in dies:
            hollow = (tech == "FeRAM")
            ax.plot(p, t, "o" if hollow else "X", ms=8.5,
                    mfc="none" if hollow else color, mec=color, mew=1.8,
                    color=color, zorder=6)

    # tier-1 (bottom, hottest die) peak callouts, colour-matched.
    # Per-series offsets so none collides: pure-DRAM up/left (clears the oval),
    # 4 FeRAM 414 W down (clears the pure-DRAM label), XBM as-is.
    CALLOUT_OFF = [(4, 12), (16, -12), (16, 1)]
    for i, (label, color, dies) in enumerate(data):
        t1 = dies[0][2]
        ax.annotate(f"{t1:.1f}°C", (1, t1), xytext=CALLOUT_OFF[i],
                    textcoords="offset points",
                    ha="left", va="bottom", fontsize=15, fontweight="bold", color=color, zorder=8)

    # hottest DRAM die of each hybrid series (its first DRAM tier above the FeRAM
    # block) — the peak-DRAM number quoted for these configs elsewhere; same
    # callout style as tier 1. Skipped for pure-DRAM, whose tier 1 is already labelled.
    # the 414 W callout is nudged right so it clears the headroom arrow at x=4.6
    DRAM_PEAK_OFF = [(20, 8), (0, 11)]
    for i, (_lbl, color, dies) in enumerate(data[1:]):
        p_d, _, t_d = next((d for d in dies if d[1] == "DRAM"))
        ax.annotate(f"{t_d:.1f}°C", (p_d, t_d), xytext=DRAM_PEAK_OFF[i],
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=15, fontweight="bold",
                    color=color, zorder=8)

    # DRAM thermal-limit cutoffs
    n_tiers = max(len(d[2]) for d in data)
    for temp_c, col in ((90.0, "#b30000"), (85.0, "#e08214")):
        ax.axhline(temp_c, ls=":", lw=2.0, color=col, alpha=0.9, zorder=2)
        ax.annotate(f"{temp_c:.0f} °C", (n_tiers + 0.42, temp_c), ha="right", va="bottom",
                    fontsize=15, fontweight="bold", color=col, zorder=8)

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
        Line2D([0], [0], lw=0, marker="o", ms=9, mfc="none", mec=INK, mew=1.8, label="FeRAM tier"),
        Line2D([0], [0], lw=0, marker="X", ms=9, mfc=INK, mec=INK, mew=1.8, label="DRAM tier"),
    ]
    leg2 = ax.legend(handles=type_handles, title="Die type", loc="upper right",
                     bbox_to_anchor=(1.0, 0.845), frameon=True, framealpha=0.95,
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
                xytext=(3.55, 101.8), ha="left", va="center",
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
    # label parked in the open space right of the curves, arrow back to the gap
    ax.annotate("thermal headroom due to\nrefresh-free FeRAM", (hx + 0.12, (h_hi + h_lo) / 2.0),
                xytext=(6.5, 96.0), ha="left", va="center",
                fontsize=13, fontweight="bold", fontstyle="italic", color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                connectionstyle="arc3,rad=-0.15"))
    fx = 2.6
    f_hi, f_lo = np.interp(fx, tiers, y_yel), np.interp(fx, tiers, y_pur)
    ax.annotate("", (fx, f_lo + 0.35), xytext=(fx, f_hi - 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#7d3c98", lw=2.2,
                                mutation_scale=18), zorder=7)
    ax.annotate("0.8× frequency\nscaling", (2.85, 76.5), ha="left", va="center",
                fontsize=14.5, fontweight="bold", color="#7d3c98", zorder=8)

    stem = "nvdram_4NV_freq_comparison"
    if with_layout:
        add_layout_inset(ax)
        stem += "_layout"
    for ext in ("png", "pdf"):
        out = os.path.join(EXP, f"{stem}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    import sys
    main(with_layout="--with-layout" in sys.argv)
