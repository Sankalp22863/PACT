"""
Heatmap plotter for the 3D HBM-on-GPU stack.
============================================
Reads the per-layer steady-state temperature grids written by PACT
(`<prefix>.grid.steady.layer{N}`) and renders them as heatmaps.

By default it tiles every device layer into one figure with a shared colour
scale (so layers are directly comparable) and overlays the four HBM-stack
footprints. Use --layer to render a single layer instead.

Usage:
    python plot_heatmap.py [--prefix dram3d_3d.grid.steady]
                           [--lcf dram3d_3d_lcf.csv]
                           [--layer N] [--out FILE] [--include-package]
                           [--gpu-side 0.024] [--hbm-side 0.010]

    --prefix P          Grid-file prefix (default: dram3d_3d.grid.steady).
    --lcf F             LCF used for the run, for layer titles (default: derived
                        from prefix, e.g. dram3d_3d_lcf.csv).
    --layer N           Plot only layer N (default: all device layers).
    --include-package   Also plot the PACT NoPackage cooling-lid layer.
    --out FILE          Output image (default: <prefix>.heatmaps.png).
    --gpu-side / --hbm-side  Geometry for the HBM overlay boxes (must match the
                        generator values used for the run).
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless / no display
import matplotlib.pyplot as plt

KELVIN = 273.15
CMAP   = "inferno"
DPI    = 150

# Map floorplan file -> friendly layer name (for titles).
FLP_NAMES = {
    "gpu_substrate_flp.csv": "GPU substrate/TSV",
    "gpu_feol_flp.csv":      "GPU FEOL (414 W source)",
    "gpu_beol_flp.csv":      "GPU BEOL",
    "ubump_flp.csv":         "uBump interface",
    "dram_tier_flp.csv":     "HBM DRAM tier",
}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "figure.dpi":    DPI,
    "savefig.dpi":   DPI,
    "savefig.bbox":  "tight",
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


def discover_layers(prefix):
    ids = []
    for p in glob.glob(f"{prefix}.layer*"):
        m = re.search(r"layer(\d+)$", p)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def layer_labels(lcf_path, n_device):
    """Build {layer_id: title}. Device-layer names come from the LCF; the single
    trailing layer PACT appends is the NoPackage cooling lid."""
    labels = {}
    if lcf_path and os.path.exists(lcf_path):
        with open(lcf_path) as fh:
            next(fh, None)  # header
            for line in fh:
                parts = [c.strip() for c in line.split(",")]
                if len(parts) < 2:
                    continue
                idx, flp = int(parts[0]), parts[1]
                labels[idx] = FLP_NAMES.get(flp, flp)
    labels[n_device] = "Cooling lid (NoPackage)"
    return labels


def hbm_extents(side, gpu_side, hbm_side):
    """Cell-index (lo, hi) ranges for the 2x2 HBM stacks along one axis."""
    gap = gpu_side - 2 * hbm_side
    margin = gap / 4.0
    central = gap / 2.0
    a0 = margin / gpu_side
    a1 = (margin + hbm_side) / gpu_side
    a2 = (margin + hbm_side + central) / gpu_side
    a3 = (margin + 2 * hbm_side + central) / gpu_side
    lo1, hi1 = int(round(a0 * side)), int(round(a1 * side))
    lo2, hi2 = int(round(a2 * side)), int(round(a3 * side))
    return (lo1, hi1), (lo2, hi2)


def _draw_hbm_boxes(ax, side, gpu_side, hbm_side):
    (lo1, hi1), (lo2, hi2) = hbm_extents(side, gpu_side, hbm_side)
    for (x0, x1) in ((lo1, hi1), (lo2, hi2)):
        for (y0, y1) in ((lo1, hi1), (lo2, hi2)):
            ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="cyan", lw=1.0, ls="--"))


def plot_single(prefix, layer_id, label, out, gpu_side, hbm_side):
    g = load_grid(prefix, layer_id)
    if g is None:
        raise SystemExit(f"No grid file for layer {layer_id} (prefix '{prefix}').")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(g, origin="lower", cmap=CMAP, aspect="equal")
    _draw_hbm_boxes(ax, g.shape[0], gpu_side, hbm_side)
    ax.set_title(f"Layer {layer_id}: {label}\n"
                 f"min {g.min():.1f}  mean {g.mean():.1f}  max {g.max():.1f} °C")
    ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="Temperature (°C)", fraction=0.046, pad=0.04)
    fig.savefig(out)
    print(f"  Written: {out}")


def plot_grid(prefix, layer_ids, labels, out, gpu_side, hbm_side):
    grids = {lid: load_grid(prefix, lid) for lid in layer_ids}
    grids = {lid: g for lid, g in grids.items() if g is not None}
    if not grids:
        raise SystemExit(f"No grid files found for prefix '{prefix}'.")
    vmin = min(g.min() for g in grids.values())
    vmax = max(g.max() for g in grids.values())

    ids = sorted(grids)
    ncols = min(4, len(ids))
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False)
    im = None
    for ax, lid in zip(axes.flat, ids):
        g = grids[lid]
        im = ax.imshow(g, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax,
                       aspect="equal")
        _draw_hbm_boxes(ax, g.shape[0], gpu_side, hbm_side)
        ax.set_title(f"L{lid}: {labels.get(lid, '?')}\nmax {g.max():.1f} °C",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"Steady-state temperature per layer — {prefix}\n"
                 "(dashed boxes = four HBM stacks; surround = thermal silicon)",
                 fontsize=13)
    fig.colorbar(im, ax=axes, label="Temperature (°C)",
                 fraction=0.015, pad=0.02)
    fig.savefig(out)
    print(f"  Written: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot PACT steady-state heatmaps for the 3D HBM-on-GPU stack.")
    p.add_argument("--prefix", default="dram3d_3d.grid.steady")
    p.add_argument("--lcf", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--include-package", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--gpu-side", type=float, default=0.024)
    p.add_argument("--hbm-side", type=float, default=0.010)
    args = p.parse_args()

    lcf = args.lcf
    if lcf is None:
        base = args.prefix.split(".grid.steady")[0]
        lcf = f"{base}_lcf.csv"

    all_ids = discover_layers(args.prefix)
    if not all_ids:
        raise SystemExit(f"No '{args.prefix}.layer*' files found. Run PACT first.")
    n_device = max(all_ids)          # PACT appends 1 NoPackage lid layer
    labels = layer_labels(lcf, n_device)

    if args.layer is not None:
        out = args.out or f"{args.prefix}.layer{args.layer}.png"
        plot_single(args.prefix, args.layer, labels.get(args.layer, "?"), out,
                    args.gpu_side, args.hbm_side)
        return

    ids = all_ids if args.include_package else [i for i in all_ids if i < n_device]
    out = args.out or f"{args.prefix}.heatmaps.png"
    plot_grid(args.prefix, ids, labels, out, args.gpu_side, args.hbm_side)


if __name__ == "__main__":
    main()
