"""
Heatmap plotter for the NVDRAM-on-GPU 3D stack.
===============================================
Reads the per-layer steady-state temperature grids written by PACT
(`<prefix>.grid.steady.layer{N}`) and renders them as heatmaps.

By default it tiles every device layer into one figure with a shared colour
scale (so layers are directly comparable) and marks the central 8.5 mm memory
macro. Use --layer to render a single layer instead.

Usage:
    python plot_heatmap.py [--prefix stacked_n4.grid.steady]
                           [--lcf stacked_n4_lcf.csv]
                           [--layer N] [--out FILE] [--include-package]

    --prefix P          Grid-file prefix (default: stacked_n4.grid.steady).
    --lcf F             LCF used for the run, for layer labels (default: derived
                        from prefix, e.g. stacked_n4_lcf.csv).
    --layer N           Plot only layer N (default: all device layers in a grid).
    --include-package   Also plot the PACT heat-spreader + heat-sink layers.
    --out FILE          Output image path (default: <prefix>.heatmaps.png or
                        <prefix>.layer{N}.png).
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
SURFACE_CMAP = "jet"   # rainbow map for the flat color-plane layer view (paper image (d))
DPI    = 150

# Map floorplan file -> friendly layer name (for titles).
FLP_NAMES = {
    "gpu_substrate_flp.csv": "GPU substrate/TSV",
    "gpu_feol_flp.csv":      "GPU FEOL/TSV",
    "gpu_beol_flp.csv":      "GPU BEOL",
    "hb_ubump_flp.csv":      "HB microbumps",
    "nvdram_beol_flp.csv":   "NVDRAM BEOL",
    "nvdram_feol_flp.csv":   "NVDRAM FEOL",
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
    """Return sorted list of layer ids that have a grid file."""
    ids = []
    for p in glob.glob(f"{prefix}.layer*"):
        m = re.search(r"layer(\d+)$", p)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def layer_labels(lcf_path, n_device):
    """Build {layer_id: title}. Device-layer names come from the LCF; the two
    trailing layers PACT appends are the heat spreader and heat sink."""
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
    # the package layers added by PACT
    labels[n_device] = "Heat Spreader"
    labels[n_device + 1] = "Heat Sink"
    return labels


def macro_extent(side, gpu_side=0.017, mem_side=0.0085):
    """Cell index range [lo, hi) of the centred memory macro, for an overlay box."""
    frac = (gpu_side - mem_side) / 2.0 / gpu_side
    lo = int(round(frac * side))
    hi = side - lo
    return lo, hi


def _draw_macro_box(ax, side):
    lo, hi = macro_extent(side)
    ax.add_patch(plt.Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                               fill=False, edgecolor="cyan", lw=1.2, ls="--"))


def plot_surface_3d(prefix, layer_id, label, out, gpu_side=0.017):
    """Render one layer as a tilted FLAT color plane (paper image (d) style).

    The plane carries no height displacement — temperature is shown purely as
    colour on a flat surface viewed in iso/perspective, like the attached
    "HBM Merged" figure. The lowest layer (layer 0, package/base side) is the
    default target; the peak temperature is annotated on the figure.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    g = load_grid(prefix, layer_id)
    if g is None:
        raise SystemExit(f"No grid file for layer {layer_id} (prefix '{prefix}').")
    side = g.shape[0]
    # Physical mm coordinates so the plane footprint matches the die size.
    extent_mm = gpu_side * 1e3
    xs = np.linspace(0, extent_mm, side)
    ys = np.linspace(0, extent_mm, side)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(g)                       # flat plane (no height bumps)

    cmap = plt.get_cmap(SURFACE_CMAP)
    norm = Normalize(vmin=g.min(), vmax=g.max())

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(g)), shade=False,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.view_init(elev=35, azim=-60)            # tilted, paper-like iso view
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_zticks([])                          # flat plane -> no temperature axis
    ax.set_box_aspect((1, 1, 0.05))            # squash z so it reads as a plane
    ax.set_title(f"Layer {layer_id}: {label} — 2D temperature map")

    peak = g.max()
    ax.text2D(0.02, 0.95, f"Peak: {peak:.1f} °C",
              transform=ax.transAxes, fontsize=13, fontweight="bold",
              color="#b30000",
              bbox=dict(boxstyle="round", fc="white", ec="#b30000", alpha=0.85))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(g)
    fig.colorbar(mappable, ax=ax, label="Temperature (°C)",
                 fraction=0.03, pad=0.10, shrink=0.6)
    fig.savefig(out)
    print(f"  Written: {out}")


def plot_single(prefix, layer_id, label, out):
    g = load_grid(prefix, layer_id)
    if g is None:
        raise SystemExit(f"No grid file for layer {layer_id} (prefix '{prefix}').")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(g, origin="lower", cmap=CMAP, aspect="equal")
    _draw_macro_box(ax, g.shape[0])
    ax.set_title(f"Layer {layer_id}: {label}\n"
                 f"min {g.min():.1f}  mean {g.mean():.1f}  max {g.max():.1f} °C")
    ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="Temperature (°C)", fraction=0.046, pad=0.04)
    fig.savefig(out)
    print(f"  Written: {out}")


def plot_grid(prefix, layer_ids, labels, out):
    grids = {lid: load_grid(prefix, lid) for lid in layer_ids}
    grids = {lid: g for lid, g in grids.items() if g is not None}
    if not grids:
        raise SystemExit(f"No grid files found for prefix '{prefix}'.")
    # shared colour scale across all shown layers
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
        _draw_macro_box(ax, g.shape[0])
        ax.set_title(f"L{lid}: {labels.get(lid, '?')}\nmax {g.max():.1f} °C",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    # hide any unused axes
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"Steady-state temperature per layer — {prefix}\n"
                 "(dashed box = 8.5 mm NVDRAM macro; surround = filler)",
                 fontsize=13)
    fig.colorbar(im, ax=axes, label="Temperature (°C)",
                 fraction=0.015, pad=0.02)
    fig.savefig(out)
    print(f"  Written: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot PACT steady-state heatmaps for the NVDRAM-on-GPU stack.")
    p.add_argument("--prefix", default="stacked_n4.grid.steady",
                   help="Grid-file prefix (default: stacked_n4.grid.steady)")
    p.add_argument("--lcf", default=None,
                   help="LCF file for layer labels (default: derived from prefix)")
    p.add_argument("--layer", type=int, default=None,
                   help="Plot only this layer (default: all device layers)")
    p.add_argument("--include-package", action="store_true",
                   help="Also plot the heat spreader + heat sink layers")
    p.add_argument("--surface3d", action="store_true",
                   help="Render a tilted flat color plane (paper image (d) style, "
                        "color = temperature, no height) of the lowest layer "
                        "(or --layer N) with the peak temperature annotated")
    p.add_argument("--out", default=None, help="Output image path")
    args = p.parse_args()

    # Derive default LCF name from prefix: 'stacked_n4.grid.steady' -> 'stacked_n4_lcf.csv'
    lcf = args.lcf
    if lcf is None:
        base = args.prefix.split(".grid.steady")[0]
        lcf = f"{base}_lcf.csv"

    all_ids = discover_layers(args.prefix)
    if not all_ids:
        raise SystemExit(f"No '{args.prefix}.layer*' files found. Run PACT first.")
    n_device = max(all_ids) - 1  # last two are spreader + sink
    labels = layer_labels(lcf, n_device)

    if args.surface3d:
        # Default to the lowest layer (layer 0, package/base side).
        lid = args.layer if args.layer is not None else min(all_ids)
        out = args.out or f"{args.prefix}.layer{lid}.surface3d.png"
        plot_surface_3d(args.prefix, lid, labels.get(lid, "?"), out)
        return

    if args.layer is not None:
        out = args.out or f"{args.prefix}.layer{args.layer}.png"
        plot_single(args.prefix, args.layer, labels.get(args.layer, "?"), out)
        return

    ids = all_ids if args.include_package else [i for i in all_ids if i < n_device]
    out = args.out or f"{args.prefix}.heatmaps.png"
    plot_grid(args.prefix, ids, labels, out)


if __name__ == "__main__":
    main()
