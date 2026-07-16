"""
φ-HBM package layout — top-down die + cross-section (drawn from the model files)
================================================================================
Renders an accurate schematic of the fully-optimized φ-HBM STCO endpoint
(5_thermal_si: base-die removed, stacks merged, top-die thinned, thermal silicon,
4 Fe-RAM + 8 DRAM) DIRECTLY from its floorplan CSVs + LCF, so the picture always
matches what PACT actually simulates:

  (a) top-down die layout — two memory stack columns on the left/right of the
      30x22 mm die, a central merged-silicon spine, thermal-silicon edge inserts.
  (b) cross-section — the vertical stack bottom (adiabatic laminate side) -> top
      (liquid cold plate), grouping the 4 Fe-RAM + 8 DRAM tiers.

Output: package_layout.png / .pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "5_thermal_si")
GPU_X, GPU_Y = 0.030, 0.022                     # die length (x) / width (y), m
INK, MUTE = "#22303a", "#8a8a8a"

# material -> (friendly name, fill colour)
MAT = {
    "NV_DIE_SI":     ("Fe-RAM die",           "#66bd63"),
    "DRAM_SI":       ("DRAM die",             "#4a90c2"),
    "MERGE_SI":      ("Merged thermal Si",    "#2c6fa6"),
    "THERMAL_SI":    ("Thermal-Si insert",    "#9ecae1"),
    "GPU_SI":        ("GPU Si substrate",     "#c6dbef"),
    "GPU_FEOL":      ("GPU FEOL (heat source)", "#e34a33"),
    "BEOL_MXY":      ("GPU BEOL (MXY)",       "#fee391"),
    "BSPDN":         ("BSPDN",                "#fdae6b"),
    "OXIDE":         ("Oxide",                "#f0f0f0"),
    "GPU_HBM_UBUMP": ("GPU–HBM µbump",        "#dadaeb"),
    "HYBRID_BOND":   ("Hybrid bond",          "#c7e9c0"),
    "NV_DIE_BEOL":   ("Fe-RAM BEOL",          "#c7e9c0"),
    "DRAM_BEOL":     ("DRAM BEOL",            "#fdd0a2"),
    "TIM":           ("TIM",                  "#fa9fb5"),
    "LID":           ("Lid (cold plate)",     "#bdbdbd"),
}
DEF = ("?", "#dddddd")


def read_flp(path):
    out = []
    for line in open(path).read().splitlines()[1:]:
        p = [c.strip() for c in line.split(",")]
        if len(p) >= 7:
            out.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]), p[6]))
    return out


def read_lcf(path):
    out = []
    for line in open(path).read().splitlines()[1:]:
        p = [c.strip() for c in line.split(",")]
        if len(p) >= 3:
            out.append((int(p[0]), p[1], float(p[2])))
    return out


def macro(flp_file):
    """Dominant (largest-area) material label of a layer's floorplan."""
    blocks = read_flp(os.path.join(RUN, flp_file))
    return max(blocks, key=lambda b: b[3] * b[4])[5]


# ── panel (a): top-down die layout ───────────────────────────────────────────
def top_down(ax):
    blocks = read_flp(os.path.join(RUN, "nv_tier_flp.csv"))   # a memory-tier layer
    mm = 1e3
    for name, x, y, l, w, label in blocks:
        col = ("#7ec87e" if name.startswith("MEM") else MAT.get(label, DEF)[1])
        ax.add_patch(Rectangle((x * mm, y * mm), l * mm, w * mm, fc=col,
                               ec="white", lw=1.4, zorder=2))
    # die outline
    ax.add_patch(Rectangle((0, 0), GPU_X * mm, GPU_Y * mm, fc="none",
                           ec=INK, lw=2.2, zorder=3))
    # labels
    ax.text(GPU_X * mm * 0.5, GPU_Y * mm * 0.5, "Merged\nthermal Si\n(central spine)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
    for cx in (GPU_X * mm * 0.19, GPU_X * mm * 0.81):
        ax.text(cx, GPU_Y * mm * 0.5, "Memory\nstack\n4 Fe-RAM\n+ 8 DRAM",
                ha="center", va="center", fontsize=9.5, fontweight="bold", color="#14431a", zorder=4)
    ax.text(GPU_X * mm * 0.5, -2.2, "← 30 mm GPU die →", ha="center", va="top",
            fontsize=10, color=INK, fontweight="bold")
    ax.text(-2.2, GPU_Y * mm * 0.5, "← 22 mm →", ha="center", va="center",
            rotation=90, fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(-4, GPU_X * mm + 4)
    ax.set_ylim(-4, GPU_Y * mm + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a)  Top-down die layout", fontsize=13, fontweight="bold", color=INK, pad=8)


# ── panel (b): cross-section ─────────────────────────────────────────────────
def cross_section(ax):
    lcf = read_lcf(os.path.join(RUN, "hybrid_lcf.csv"))
    n_nv = sum(1 for _, f, _ in lcf if f == "nv_tier_flp.csv")
    n_dram = sum(1 for _, f, _ in lcf if f == "dram_tier_flp.csv")
    # build display bands bottom -> top
    bands, seen_nv, seen_dram = [], False, False
    for lid, flp, th in lcf:
        if flp in ("hybrid_bond_flp.csv", "nv_die_beol_flp.csv", "dram_die_beol_flp.csv"):
            continue                                  # sublayers folded into the tier band
        if flp == "nv_tier_flp.csv":
            if not seen_nv:
                bands.append((f"{n_nv}× Fe-RAM die  (refresh-free · immutable weights)",
                              "NV_DIE_SI", 2.4)); seen_nv = True
            continue
        if flp == "dram_tier_flp.csv":
            if not seen_dram:
                bands.append((f"{n_dram}× DRAM die  (mutable data)", "DRAM_SI", 3.0)); seen_dram = True
            continue
        name = MAT.get(macro(flp), DEF)[0]
        h = 1.5 if flp == "gpu_si_flp.csv" else (1.2 if flp in ("lid_flp.csv", "tim_flp.csv") else 0.85)
        thick = f"{th*1e6:g} µm"
        bands.append((f"{name}   ·   {thick}", macro(flp), h))

    y = 0.0
    total_h = sum(h for _, _, h in bands)
    for label, m, h in bands:
        col = MAT.get(m, DEF)[1]
        hot = (m == "GPU_FEOL")
        ax.add_patch(Rectangle((0.06, y), 0.88, h, fc=col, ec=INK,
                               lw=1.8 if hot else 0.9, zorder=2))
        ax.text(0.5, y + h / 2.0, label, ha="center", va="center",
                fontsize=9.6, fontweight="bold" if (hot or m in ("NV_DIE_SI", "DRAM_SI")) else "normal",
                color="white" if m in ("GPU_FEOL", "NV_DIE_SI", "DRAM_SI", "MERGE_SI") else INK, zorder=3)
        y += h

    # top cooling arrows (liquid cold plate) + bottom adiabatic laminate
    for xa in np.linspace(0.16, 0.84, 6):
        ax.add_patch(FancyArrowPatch((xa, y + 0.9), (xa, y + 0.15),
                     arrowstyle="-|>", mutation_scale=14, color="#2c7fb8", lw=2.0, zorder=4))
    ax.text(0.5, y + 1.05, "Liquid cold plate   (top-side, HTC = 30,000 W/m²K)",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1f6f8b")
    ax.add_patch(Rectangle((0.06, -0.9), 0.88, 0.55, fc="#efe6d8", ec=INK, lw=0.9, hatch="////", zorder=2))
    ax.text(0.5, -0.62, "Laminate / package side  (bottom — adiabatic in this model)",
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color=INK)
    ax.annotate("", (0.02, 0.0), (0.02, y), arrowprops=dict(arrowstyle="<-", color=MUTE, lw=1.4))
    ax.text(0.005, y / 2.0, "stack ↑ to cooling", rotation=90, ha="right", va="center",
            fontsize=8.5, color=MUTE)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-1.1, y + 1.7)
    ax.axis("off")
    ax.set_title("(b)  Cross-section  (fully STCO-optimized φ-HBM)", fontsize=13,
                 fontweight="bold", color=INK, pad=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})
    fig.patch.set_facecolor("white")
    top_down(axes[0])
    cross_section(axes[1])
    fig.suptitle("φ-HBM package layout — fully STCO-optimized (4 Fe-RAM + 8 DRAM, base die removed)",
                 fontsize=15, fontweight="bold", color=INK, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.03, wspace=0.05)
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"package_layout.{ext}")
        fig.savefig(out, dpi=200, facecolor="white")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
