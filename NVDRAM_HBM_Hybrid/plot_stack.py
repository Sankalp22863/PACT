"""
Stack schematic + layer-properties table + future-integration diagram.
======================================================================
Reproduces three of the paper figures directly from a PACT experiment's LCF and
experiment.config (so it stays in sync with whatever was simulated):

  * stack_cross_section.png  -- cross-section schematic (paper image (b)): the
    layers stacked bottom->top, drawn with memory tiers as two side-by-side
    memory columns separated by a central thermal-silicon column (memory on both
    sides with thermal silicon in the middle).
  * stack_layer_table.png    -- layer-properties table (paper image (c)-table):
    one row per distinct stack layer with thickness, in-plane and cross-plane
    thermal conductivity (k = 1 / thermalresistivity from the config).
  * future_integration.png   -- high-level "future 3D integration" block diagram
    (last paper image): GPU -> NVDRAM stack (bottom) -> DRAM stack (top) -> lid.

Works for any of the repo's stacks; defaults target the hybrid experiment:
    python3 plot_stack.py [--lcf hybrid_lcf.csv] [--config experiment.config]
                          [--flp-dir .] [--out-prefix stack]
"""

import argparse
import configparser
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DPI = 150
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "figure.dpi":   DPI,
    "savefig.dpi":  DPI,
    "savefig.bbox": "tight",
})

# Surround/filler materials are drawn as the "central column" in memory tiers.
SURROUND = {"THERMAL_SI", "FILLER"}

# Friendly names + colours per material label.
MAT_NAME = {
    "GPU_Si":     "GPU substrate (Si)",
    "GPU_FEOL":   "GPU FEOL",
    "GPU_BEOL":   "GPU BEOL",
    "UBUMP":      "GPU-mem uBump",
    "HB_uBump":   "Hybrid-bond uBump",
    "NV_FEOL":    "NVDRAM die",
    "NV_BEOL":    "NVDRAM BEOL",
    "DRAM_Si":    "DRAM die",
    "THERMAL_SI": "Thermal silicon",
    "DAF":        "Die-attach film",
    "FILLER":     "Dielectric filler",
}
MAT_COLOR = {
    "GPU_Si":     "#9ecae1",
    "GPU_FEOL":   "#fb6a4a",
    "GPU_BEOL":   "#fcbba1",
    "UBUMP":      "#dadaeb",
    "HB_uBump":   "#dadaeb",
    "NV_FEOL":    "#74c476",
    "NV_BEOL":    "#c7e9c0",
    "DRAM_Si":    "#6baed6",
    "THERMAL_SI": "#3182bd",
    "DAF":        "#fdd0a2",
    "FILLER":     "#d9d9d9",
}
DEFAULT_COLOR = "#cccccc"


def read_lcf(lcf_path):
    """Return [(layer_id, floorplan_file, thickness_m), ...] in stack order."""
    rows = []
    with open(lcf_path) as fh:
        next(fh, None)
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]), parts[1], float(parts[2])))
    return rows


def read_flp_labels(flp_path):
    """Return the ordered list of (block_name, label) for a floorplan."""
    out = []
    with open(flp_path) as fh:
        next(fh, None)
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 7:
                continue
            out.append((parts[0], parts[6]))
    return out


def layer_materials(flp_path):
    """Classify a layer floorplan -> (kind, macro_label, surround_label).

    kind == "full"   : single full-die material (macro_label set, surround None)
    kind == "tier"   : memory tier (macro on both sides, surround in the middle)
    """
    labels = read_flp_labels(flp_path)
    uniq = []
    for _, lab in labels:
        if lab not in uniq:
            uniq.append(lab)
    surround = [l for l in uniq if l in SURROUND]
    macro = [l for l in uniq if l not in SURROUND]
    if len(uniq) == 1:
        return "full", uniq[0], None
    macro_label = macro[0] if macro else uniq[0]
    surround_label = surround[0] if surround else None
    return "tier", macro_label, surround_label


def material_k(config_path):
    """Return {label: k_W_mK} from thermalresistivity in experiment.config."""
    cp = configparser.ConfigParser()
    cp.read(config_path)
    ks = {}
    for sec in cp.sections():
        key = "thermalresistivity ((m-k)/w)"
        if cp.has_option(sec, key):
            rho = cp.getfloat(sec, key)
            if rho > 0:
                ks[sec] = 1.0 / rho
    return ks


# ── Cross-section schematic (image b) ───────────────────────────────────────
def draw_cross_section(layers, flp_dir, ks, out):
    n = len(layers)
    # box heights ∝ log(thickness) so 100 µm and 1 µm both stay visible
    th = np.array([t for _, _, t in layers])
    h = np.log10(th / th.min()) + 1.0
    h = h / h.sum()                       # normalise to total 1.0 in axis units

    fig, ax = plt.subplots(figsize=(8.5, max(7, 0.42 * n)))
    y = 0.0
    used = {}                             # label -> color (for legend)
    for (lid, flp, t), height in zip(layers, h):
        kind, macro, surround = layer_materials(os.path.join(flp_dir, flp))
        if kind == "full":
            col = MAT_COLOR.get(macro, DEFAULT_COLOR)
            ax.add_patch(plt.Rectangle((0.05, y), 0.90, height,
                                       fc=col, ec="black", lw=0.6))
            used[macro] = col
        else:
            mcol = MAT_COLOR.get(macro, DEFAULT_COLOR)
            scol = MAT_COLOR.get(surround, DEFAULT_COLOR)
            # left memory column | central thermal-silicon | right memory column
            ax.add_patch(plt.Rectangle((0.05, y), 0.34, height, fc=mcol, ec="black", lw=0.6))
            ax.add_patch(plt.Rectangle((0.39, y), 0.17, height, fc=scol, ec="black", lw=0.6))
            ax.add_patch(plt.Rectangle((0.56, y), 0.34, height, fc=mcol, ec="black", lw=0.6))
            used[macro] = mcol
            if surround:
                used[surround] = scol
        # right-hand label
        ax.text(0.965, y + height / 2.0,
                f"L{lid} {MAT_NAME.get(macro, macro)}  ({t*1e6:g} µm)",
                va="center", ha="left", fontsize=7.5)
        y += height

    # conceptual top lid + bottom package (context; PACT adds the convective lid)
    ax.add_patch(FancyBboxPatch((0.05, y + 0.005), 0.85, 0.03,
                                boxstyle="round,pad=0.002", fc="#bdd7e7", ec="black"))
    ax.text(0.475, y + 0.02, "Liquid cold-plate lid (top cooling)",
            ha="center", va="center", fontsize=8, style="italic")
    ax.add_patch(FancyBboxPatch((0.05, -0.035), 0.85, 0.03,
                                boxstyle="round,pad=0.002", fc="#d9d9d9", ec="black"))
    ax.text(0.475, -0.02, "Laminate / package (adiabatic bottom)",
            ha="center", va="center", fontsize=8, style="italic")

    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black")
               for c in used.values()]
    txt = [f"{MAT_NAME.get(l, l)}  (k={ks.get(l, float('nan')):.0f} W/m·K)"
           for l in used]
    ax.legend(handles, txt, loc="center left", bbox_to_anchor=(-0.55, 0.5),
              fontsize=7.5, frameon=True)

    ax.set_xlim(-0.05, 1.35)
    ax.set_ylim(-0.06, y + 0.06)
    ax.axis("off")
    ax.set_title("Cross-section stack (bottom→top)\n"
                 "memory tiers: memory | thermal-silicon | memory",
                 fontsize=12)
    fig.savefig(out)
    print(f"  Written: {out}")


# ── Layer-properties table (image c-table) ──────────────────────────────────
def draw_table(layers, flp_dir, ks, out):
    # collapse consecutive identical floorplans into one row with a ×count
    rows = []
    for lid, flp, t in layers:
        kind, macro, _ = layer_materials(os.path.join(flp_dir, flp))
        name = MAT_NAME.get(macro, macro)
        k = ks.get(macro, float("nan"))
        if rows and rows[-1][0] == name and abs(rows[-1][1] - t) < 1e-15:
            rows[-1][3] += 1
        else:
            rows.append([name, t, k, 1])

    cell_text, total_t = [], 0.0
    for name, t, k, cnt in rows:
        total_t += t * cnt * 1e6
        label = f"{name}" + (f"  ×{cnt}" if cnt > 1 else "")
        cell_text.append([label, f"{t*1e6:g}", f"{k:.1f}", f"{k:.1f}"])
    cell_text.append(["TOTAL device stack", f"{total_t:g}", "—", "—"])

    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.34 * (len(cell_text) + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Stack layer (bottom→top)", "Thickness (µm)",
                   "T.C. in-plane (W/m·K)", "T.C. cross-plane (W/m·K)"],
        colWidths=[0.34, 0.20, 0.23, 0.23],
        cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    # header styling + total row
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#3182bd"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(cell_text):
            cell.set_facecolor("#deebf7"); cell.set_text_props(fontweight="bold")
    ax.set_title("Cross-section stack layers — thickness & thermal conductivity",
                 fontsize=12, pad=12)
    fig.savefig(out)
    print(f"  Written: {out}")


# ── Future-integration block diagram (last image) ───────────────────────────
def draw_future_integration(layers, flp_dir, out):
    # count NVDRAM vs DRAM tiers from the LCF
    nv = dram = 0
    for _, flp, _ in layers:
        kind, macro, _ = layer_materials(os.path.join(flp_dir, flp))
        if macro == "NV_FEOL":
            nv += 1
        elif macro == "DRAM_Si":
            dram += 1

    blocks = [
        ("Package / Laminate", "#d9d9d9"),
        ("GPU compute die (heat source)", "#fb6a4a"),
        ("uBump / hybrid-bond interface", "#dadaeb"),
        (f"NVDRAM stack — {nv} tiers (non-volatile, bottom)", "#74c476"),
        (f"DRAM stack — {dram} tiers (HBM, top)", "#6baed6"),
        ("Liquid cold-plate lid (cooling)", "#bdd7e7"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    y = 0.0
    bw, bh, gap = 0.8, 0.10, 0.045
    for i, (label, col) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((0.1, y), bw, bh,
                                    boxstyle="round,pad=0.008",
                                    fc=col, ec="black", lw=1.0))
        ax.text(0.5, y + bh / 2.0, label, ha="center", va="center",
                fontsize=10, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", xy=(0.5, y + bh + gap), xytext=(0.5, y + bh),
                        arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.6))
        y += bh + gap
    # heat-flow caption
    ax.text(0.5, -0.04, "heat flows UP to the lid →", ha="center",
            fontsize=9, style="italic", color="#b30000")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.07, y + 0.02)
    ax.axis("off")
    ax.set_title("Future 3D integration: NVDRAM (bottom) + DRAM (top) on GPU",
                 fontsize=12)
    fig.savefig(out)
    print(f"  Written: {out}")


def main():
    p = argparse.ArgumentParser(
        description="Draw stack cross-section, layer table, and future-integration diagram.")
    p.add_argument("--lcf", default="hybrid_lcf.csv")
    p.add_argument("--config", default="experiment.config")
    p.add_argument("--flp-dir", default=None,
                   help="Directory holding the floorplan CSVs (default: LCF's dir)")
    p.add_argument("--out-prefix", default="stack")
    p.add_argument("--no-future", action="store_true",
                   help="Skip the future-integration block diagram")
    args = p.parse_args()

    if not os.path.exists(args.lcf):
        raise SystemExit(f"LCF not found: {args.lcf}")
    flp_dir = args.flp_dir or (os.path.dirname(os.path.abspath(args.lcf)))
    layers = read_lcf(args.lcf)
    ks = material_k(args.config)

    draw_cross_section(layers, flp_dir, ks, f"{args.out_prefix}_cross_section.png")
    draw_table(layers, flp_dir, ks, f"{args.out_prefix}_layer_table.png")
    if not args.no_future:
        draw_future_integration(layers, flp_dir, f"{args.out_prefix}_future_integration.png")


if __name__ == "__main__":
    main()
