"""
Peak temperature up the memory stack: hybrid (NVDRAM+DRAM) vs pure DRAM.
======================================================================
Reproduces the paper's "Peak Temperature of HBM DRAM Dies" line plot (image (c))
as a per-tier profile of two integration options:

    * NVDRAM + DRAM hybrid stack  -- the NVDRAM_HBM_Hybrid run: NVDRAM tiers at
      the BOTTOM (nearest the GPU), with the DRAM tiers CONTINUING on TOP. One
      continuous curve up the whole memory stack.
    * Pure DRAM stack             -- the DRAM_3D_Stacked run (HBM only), as the
      reference.

For each run we walk the LCF, pick out the memory-die layers in stack order,
load each layer's steady-state grid and take its peak temperature. Tier index 1
is the bottom tier (nearest the GPU); the curve climbs the stack toward the lid.

Both runs use the same 414 W GPU, so the curves are directly comparable.

Usage:
    python3 plot_peak_per_tier.py [--out FILE]
        [--hybrid-dir .] [--hybrid-prefix hybrid.grid.steady] [--hybrid-lcf hybrid_lcf.csv]
        [--dram-dir ../DRAM_3D_Stacked] [--dram-prefix dram3d_3d.grid.steady]
        [--dram-lcf dram3d_3d_lcf.csv]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KELVIN = 273.15
DPI = 150

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "figure.dpi":   DPI,
    "savefig.dpi":  DPI,
    "savefig.bbox": "tight",
})


def k2c(v):
    return np.asarray(v, dtype=float) - KELVIN


def load_grid(prefix, layer_id):
    """Load one layer grid as a square °C array, or None if missing."""
    path = f"{prefix}.layer{layer_id}"
    if not os.path.exists(path):
        return None
    data = k2c(np.loadtxt(path))
    side = int(round(np.sqrt(data.size)))
    return data.reshape(side, side)


def read_lcf_layers(lcf_path):
    """Return [(layer_id, floorplan_file), ...] in stack order from an LCF."""
    rows = []
    with open(lcf_path) as fh:
        next(fh, None)  # header
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 2:
                continue
            rows.append((int(parts[0]), parts[1]))
    return rows


def stack_profile(exp_dir, prefix, lcf, flp_regions):
    """Peak °C per memory-die tier, bottom->top, for one run.

    flp_regions maps a memory floorplan file -> region name (e.g.
    {"nv_tier_flp.csv": "NVDRAM", "dram_tier_flp.csv": "DRAM"}). Every LCF layer
    using one of those floorplans contributes one tier. Returns a list of
    (region, peak_C) in stack order.
    """
    lcf_path = os.path.join(exp_dir, lcf)
    grid_prefix = os.path.join(exp_dir, prefix)
    if not os.path.exists(lcf_path):
        raise SystemExit(f"LCF not found: {lcf_path}")
    out = []
    for layer_id, flp in read_lcf_layers(lcf_path):
        if flp not in flp_regions:
            continue
        g = load_grid(grid_prefix, layer_id)
        if g is None:
            raise SystemExit(
                f"Missing grid '{grid_prefix}.layer{layer_id}'. Run PACT for "
                f"{exp_dir} first.")
        out.append((flp_regions[flp], g.max()))
    if not out:
        raise SystemExit(f"No memory-die layers found in {lcf_path}.")
    return out


def main():
    p = argparse.ArgumentParser(
        description="Peak temperature up the stack: NVDRAM+DRAM hybrid vs pure DRAM.")
    p.add_argument("--hybrid-dir", default=".")
    p.add_argument("--hybrid-prefix", default="hybrid.grid.steady")
    p.add_argument("--hybrid-lcf", default="hybrid_lcf.csv")
    p.add_argument("--dram-dir", default="../DRAM_3D_Stacked")
    p.add_argument("--dram-prefix", default="dram3d_3d.grid.steady")
    p.add_argument("--dram-lcf", default="dram3d_3d_lcf.csv")
    p.add_argument("--out", default="peak_per_tier_hybrid_vs_dram.png")
    args = p.parse_args()

    hybrid = stack_profile(args.hybrid_dir, args.hybrid_prefix, args.hybrid_lcf,
                           {"nv_tier_flp.csv": "NVDRAM", "dram_tier_flp.csv": "DRAM"})
    pure = stack_profile(args.dram_dir, args.dram_prefix, args.dram_lcf,
                         {"dram_tier_flp.csv": "DRAM"})

    regions = [r for r, _ in hybrid]
    yh = np.array([v for _, v in hybrid])
    xh = np.arange(1, len(yh) + 1)
    n_nv = sum(1 for r in regions if r == "NVDRAM")

    yp = np.array([v for _, v in pure])
    xp = np.arange(1, len(yp) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # --- hybrid stack: ONE continuous line, NVDRAM bottom -> DRAM top ---
    ax.plot(xh, yh, "-", color="#7b3294", lw=2.2, zorder=2,
            label="NVDRAM + DRAM hybrid stack")
    nv_idx = [i for i, r in enumerate(regions) if r == "NVDRAM"]
    dr_idx = [i for i, r in enumerate(regions) if r == "DRAM"]
    ax.plot(xh[nv_idx], yh[nv_idx], "o", color="#d62728", ms=8, zorder=3,
            label="   ↳ NVDRAM tier (bottom)")
    ax.plot(xh[dr_idx], yh[dr_idx], "s", color="#e6862e", ms=7, zorder=3,
            label="   ↳ DRAM tier (continues on top)")

    # shade + mark the NVDRAM region of the hybrid stack
    ax.axvspan(0.5, n_nv + 0.5, color="#d62728", alpha=0.06, zorder=0)
    ax.axvline(n_nv + 0.5, color="#888888", ls=":", lw=1.2, zorder=1)
    ylo, yhi = ax.get_ylim()
    ax.text(n_nv / 2.0 + 0.5, ylo + 0.04 * (yhi - ylo), "NVDRAM (bottom)",
            ha="center", va="bottom", fontsize=8, color="#d62728")

    # --- pure DRAM stack reference ---
    ax.plot(xp, yp, "s--", color="#1f77b4", lw=2, ms=6, zorder=2,
            label="Pure DRAM stack (HBM only)")

    # annotate the bottom-tier peak of each curve
    ax.annotate(f"{yh[0]:.1f} °C", (xh[0], yh[0]), textcoords="offset points",
                xytext=(4, 6), color="#7b3294", fontsize=9, fontweight="bold")
    ax.annotate(f"{yp[0]:.1f} °C", (xp[0], yp[0]), textcoords="offset points",
                xytext=(4, -14), color="#1f77b4", fontsize=9, fontweight="bold")

    ax.set_xlabel("Memory tier index (1 = bottom, nearest GPU → top = lid)")
    ax.set_ylabel("Peak die temperature (°C)")
    ax.set_title("Peak temperature up the memory stack\n"
                 "NVDRAM+DRAM hybrid vs pure DRAM (both at 414 W GPU)", fontsize=12)
    ax.set_xticks(np.arange(1, max(len(yh), len(yp)) + 1))
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    fig.savefig(args.out)
    print(f"  Written: {args.out}")
    print(f"  Hybrid ({n_nv} NVDRAM + {len(yh)-n_nv} DRAM): "
          + ", ".join(f"{v:.1f}" for v in yh))
    print(f"  Pure DRAM ({len(yp)}): " + ", ".join(f"{v:.1f}" for v in yp))


if __name__ == "__main__":
    main()
