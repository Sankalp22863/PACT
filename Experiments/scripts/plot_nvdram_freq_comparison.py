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
  * 4 NV + 8 DRAM  (phi-HBM, 0.7x freq)  = ../../NVDRAM_waterfall3/5_thermal_si
      STCO endpoint with 0.7x GPU frequency scaling (355.49 W -> ~297 W)

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
    ("4 NV + 8 DRAM  — φ-HBM, 0.7× freq scaling",
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

    fig, ax = plt.subplots(figsize=(11.0, 7.2))

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
        ax.annotate(f"{t1:.1f}°C", (1, t1), xytext=(7, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, fontweight="bold", color=color, zorder=8)

    n_tiers = max(len(d[2]) for d in data)
    ax.set_xticks(range(1, n_tiers + 1))
    ax.set_xticklabels([str(i) for i in range(1, n_tiers + 1)], fontsize=9.5, color=INK)
    ax.set_xlabel("Memory die position   (Base = nearest GPU  →  top tier = nearest cooling lid)",
                  fontsize=11, color=INK)
    ax.set_ylabel("Peak die temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(0.5, n_tiers + 0.5)
    ax.grid(True, color=MUTE, alpha=0.3, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    # legend 1 — configuration (colours)
    comp_handles = [Line2D([0], [0], color=color, lw=2.4, label=label) for label, color, _ in data]
    leg1 = ax.legend(handles=comp_handles, title="Configuration", loc="upper right",
                     frameon=True, fontsize=9.5, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)

    # legend 2 — die type (marker shapes)
    type_handles = [
        Line2D([0], [0], color=MUTE, lw=0, marker="o", ms=9, mfc="#bbb", mec="white", label="NVDRAM tier"),
        Line2D([0], [0], color=MUTE, lw=0, marker="s", ms=8, mfc="#bbb", mec="white", label="DRAM tier"),
    ]
    ax.legend(handles=type_handles, title="Die type", loc="lower left",
              frameon=True, fontsize=9.5, title_fontsize=10, framealpha=0.95)

    ax.set_title("Peak temperature of each memory die up the 3D stack\n"
                 "4 NVDRAM + 8 DRAM: effect of GPU frequency scaling (vs pure-DRAM baseline)",
                 fontsize=13, color=INK, pad=12)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(EXP, f"nvdram_4NV_freq_comparison.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
