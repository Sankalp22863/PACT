"""
Heatmap plotter for the hybrid NVDRAM-bottom / DRAM-top 3D stack.
================================================================
Reads the per-layer steady-state temperature grids written by PACT
(`<prefix>.grid.steady.layer{N}`) and renders them as heatmaps.

By default it tiles every device layer into one figure with a shared colour
scale (so layers are directly comparable) and overlays the four memory-stack
footprints. Use --layer to render a single layer, or --surface3d to render a
tilted 3D temperature surface (paper image (d) style) of the lowest layer.

Usage:
    python3 plot_heatmap.py [--prefix hybrid.grid.steady] [--lcf hybrid_lcf.csv]
                            [--layer N] [--surface3d] [--out FILE]
                            [--include-package] [--gpu-side 0.024] [--mem-side 0.010]
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
    "gpu_feol_flp.csv":      "GPU FEOL (heat source)",
    "gpu_beol_flp.csv":      "GPU BEOL",
    "ubump_flp.csv":         "uBump interface",
    "nv_tier_flp.csv":       "NVDRAM tier",
    "dram_tier_flp.csv":     "DRAM tier",
    "daf_flp.csv":           "DAF bond",
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


def mem_extents(side, gpu_side, mem_side):
    """Cell-index (lo, hi) ranges for the 2x2 memory stacks along one axis."""
    gap = gpu_side - 2 * mem_side
    margin = gap / 4.0
    central = gap / 2.0
    a0 = margin / gpu_side
    a1 = (margin + mem_side) / gpu_side
    a2 = (margin + mem_side + central) / gpu_side
    a3 = (margin + 2 * mem_side + central) / gpu_side
    lo1, hi1 = int(round(a0 * side)), int(round(a1 * side))
    lo2, hi2 = int(round(a2 * side)), int(round(a3 * side))
    return (lo1, hi1), (lo2, hi2)


def _draw_mem_boxes(ax, side, gpu_side, mem_side):
    (lo1, hi1), (lo2, hi2) = mem_extents(side, gpu_side, mem_side)
    for (x0, x1) in ((lo1, hi1), (lo2, hi2)):
        for (y0, y1) in ((lo1, hi1), (lo2, hi2)):
            ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="cyan", lw=1.0, ls="--"))


def plot_single(prefix, layer_id, label, out, gpu_side, mem_side):
    g = load_grid(prefix, layer_id)
    if g is None:
        raise SystemExit(f"No grid file for layer {layer_id} (prefix '{prefix}').")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(g, origin="lower", cmap=CMAP, aspect="equal")
    _draw_mem_boxes(ax, g.shape[0], gpu_side, mem_side)
    ax.set_title(f"Layer {layer_id}: {label}\n"
                 f"min {g.min():.1f}  mean {g.mean():.1f}  max {g.max():.1f} °C")
    ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
    fig.colorbar(im, ax=ax, label="Temperature (°C)", fraction=0.046, pad=0.04)
    fig.savefig(out)
    print(f"  Written: {out}")


def plot_surface_3d(prefix, layer_id, label, out, gpu_side):
    """Render one layer as a tilted FLAT color plane (paper image (d) style):
    temperature shown purely as colour on a flat iso-view plane (no height), with
    the peak annotated. Defaults to the lowest layer (package side)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    g = load_grid(prefix, layer_id)
    if g is None:
        raise SystemExit(f"No grid file for layer {layer_id} (prefix '{prefix}').")
    side = g.shape[0]
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
    ax.view_init(elev=35, azim=-60)
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


def plot_grid(prefix, layer_ids, labels, out, gpu_side, mem_side):
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
        _draw_mem_boxes(ax, g.shape[0], gpu_side, mem_side)
        ax.set_title(f"L{lid}: {labels.get(lid, '?')}\nmax {g.max():.1f} °C",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    fig.suptitle(f"Steady-state temperature per layer — {prefix}\n"
                 "(dashed boxes = four memory stacks; surround = thermal silicon)",
                 fontsize=13)
    fig.colorbar(im, ax=axes, label="Temperature (°C)",
                 fraction=0.015, pad=0.02)
    fig.savefig(out)
    print(f"  Written: {out}")


def main():
    p = argparse.ArgumentParser(description="Plot PACT steady-state heatmaps for the hybrid 3D stack.")
    p.add_argument("--prefix", default="hybrid.grid.steady")
    p.add_argument("--lcf", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--surface3d", action="store_true",
                   help="Render a tilted flat color plane (color = temperature, no "
                        "height) of the lowest layer (or --layer N)")
    p.add_argument("--include-package", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--gpu-side", type=float, default=0.024)
    p.add_argument("--mem-side", type=float, default=0.010)
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

    if args.surface3d:
        lid = args.layer if args.layer is not None else min(all_ids)
        out = args.out or f"{args.prefix}.layer{lid}.surface3d.png"
        plot_surface_3d(args.prefix, lid, labels.get(lid, "?"), out, args.gpu_side)
        return

    if args.layer is not None:
        out = args.out or f"{args.prefix}.layer{args.layer}.png"
        plot_single(args.prefix, args.layer, labels.get(args.layer, "?"), out,
                    args.gpu_side, args.mem_side)
        return

    ids = all_ids if args.include_package else [i for i in all_ids if i < n_device]
    out = args.out or f"{args.prefix}.heatmaps.png"
    plot_grid(args.prefix, ids, labels, out, args.gpu_side, args.mem_side)


if __name__ == "__main__":
    main()
