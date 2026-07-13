"""
Layerwise stack temperature profile — all-DRAM vs φ-HBM  (post-thermal-silicon)
================================================================================
For the FINAL (thermal-silicon) stage only, plot the peak temperature of every
memory die as a function of its position in the stack (die 1 = bottom, nearest
the GPU; die 12 = top, nearest the heat spreader), for:
  * all-DRAM stack  (../NVDRAM_baseline3, 12 DRAM)              — red
  * φ-HBM stack     (this folder, 4 NVDRAM bottom + 8 DRAM top) — blue
The GPU compute-die peak is shown at position 0 as the thermal anchor. Each die
temperature is the peak under the stack columns (same mask as the waterfall).
Reads only <stack>/5_thermal_si/. Output: layerwise_thermal_si.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "NVDRAM_baseline3")
STAGE = "5_thermal_si"                      # final optimization only
PREFIX = "hybrid.grid.steady"
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005    # under-stack mask geometry (matches run_waterfall)

AD, NV, DR = "#c0392b", "#08519c", "#2c7fb8"   # all-DRAM red, NVDRAM dark blue, DRAM steel blue
INK, MUTE = "#2b2b2b", "#8a8a8a"


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


def grid(stack_dir, layer):
    g = (np.loadtxt(os.path.join(stack_dir, STAGE, f"{PREFIX}.layer{layer}")) - 273.15)
    return g.reshape(ROWS, COLS)


def profile(stack_dir):
    """Return (gpu_peak, [(die_pos, tech, peak_C), ...]) for the memory dies,
    ordered bottom (nearest GPU) -> top."""
    lcf = os.path.join(stack_dir, STAGE, "hybrid_lcf.csv")
    mem_layers, gpu_layer = [], None
    with open(lcf) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "gpu_feol_flp.csv":
                gpu_layer = int(idx)
            elif flp == "nv_tier_flp.csv":
                mem_layers.append((int(idx), "NVDRAM"))
            elif flp == "dram_tier_flp.csv":
                mem_layers.append((int(idx), "DRAM"))
    mem_layers.sort()
    gpu_peak = float(grid(stack_dir, gpu_layer).max())
    dies = [(pos + 1, tech, float(grid(stack_dir, lyr)[MASK].max()))
            for pos, (lyr, tech) in enumerate(mem_layers)]
    return gpu_peak, dies


def main():
    ad_gpu, ad = profile(BASE)          # all-DRAM
    ph_gpu, ph = profile(HERE)          # φ-HBM

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    # NVDRAM region shading (φ-HBM bottom dies); y in axes fraction so it stays put
    n_nv = sum(1 for _, t, _ in ph if t == "NVDRAM")
    ax.axvspan(0.5, n_nv + 0.5, color=NV, alpha=0.06, zorder=0)
    ax.text(n_nv / 2 + 0.5, 0.965, "NVDRAM dies\n(φ-HBM bottom)",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=9, color=NV, fontweight="bold")

    # all-DRAM line (positions 0..12 with GPU at 0)
    xad = [0] + [p for p, _, _ in ad]
    yad = [ad_gpu] + [t for _, _, t in ad]
    ax.plot(xad, yad, "-o", color=AD, lw=2.4, ms=7, mec="white", mew=1.2,
            zorder=5, label="All-DRAM stack (12 DRAM)")

    # φ-HBM line, NVDRAM vs DRAM markers
    xph = [0] + [p for p, _, _ in ph]
    yph = [ph_gpu] + [t for _, _, t in ph]
    ax.plot(xph, yph, "-", color=DR, lw=2.4, zorder=5, label="φ-HBM stack (4 NVDRAM + 8 DRAM)")
    for p, tech, t in ph:
        ax.plot(p, t, "o", ms=8, mec="white", mew=1.2, zorder=6,
                color=NV if tech == "NVDRAM" else DR)
    ax.plot(0, ph_gpu, "o", ms=8, color=DR, mec="white", mew=1.2, zorder=6)

    # GPU anchor
    ax.axvline(0.5, color=MUTE, lw=0.8, ls=":", zorder=1)
    ax.annotate("GPU\ncompute die", (0, max(ad_gpu, ph_gpu)), xytext=(0, 12),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
    for x, y, c in [(0, ad_gpu, AD), (0, ph_gpu, DR)]:
        ax.annotate(f"{y:.1f}", (x, y), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8.5, color=c, fontweight="bold")

    # endpoint labels for the memory dies (bottom + top)
    for dies, col in [(ad, AD), (ph, DR)]:
        for p, tech, t in (dies[0], dies[-1]):
            ax.annotate(f"{t:.1f}", (p, t), xytext=(0, -14 if col == DR else 11),
                        textcoords="offset points", ha="center",
                        va="top" if col == DR else "bottom",
                        fontsize=8.5, color=col, fontweight="bold")

    ax.set_xticks(range(0, 13))
    ax.set_xticklabels(["GPU"] + [str(i) for i in range(1, 13)], fontsize=9.5, color=INK)
    ax.set_xlabel("Stack position   (1 = bottom die, nearest GPU  →  12 = top die, nearest spreader)",
                  fontsize=11, color=INK)
    ax.set_ylabel("Peak temperature (°C)", fontsize=12, color=INK)
    ax.set_xlim(-0.6, 12.6)
    lo = min(min(t for _, _, t in ad), min(t for _, _, t in ph))
    hi = max(ad_gpu, ph_gpu)
    ax.set_ylim(lo - 4, hi + 5)
    ax.grid(axis="y", color=MUTE, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(colors=MUTE)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK)

    handles = [
        Line2D([0], [0], color=AD, lw=2.4, marker="o", mec="white", label="All-DRAM stack (12 DRAM)"),
        Line2D([0], [0], color=DR, lw=2.4, marker="o", mec="white", label="φ-HBM stack (4 NVDRAM + 8 DRAM)"),
        Line2D([0], [0], color=NV, lw=0, marker="o", mec="white", label="NVDRAM die (φ-HBM bottom)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9.5, framealpha=0.95)

    ax.set_title("Layerwise stack temperature after thermal-silicon optimization — "
                 "all-DRAM vs φ-HBM (30 × 22 mm)\n"
                 "Per-die peak under the stacks, active read/write "
                 "(α$_w$=0.24, 4.4 TB/s/stack, 70/90 fJ/bit)",
                 fontsize=12, color=INK, pad=14)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"layerwise_thermal_si.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
