"""
Shared plotting library for the PACT 3D-stack experiments.
==========================================================
Reads the per-layer steady-state temperature grids PACT writes
(`<prefix>.grid.steady.layer{N}`) plus the experiment's LCF / experiment.config
and renders the figures used by both experiments:

  * heatmaps          - per-layer temperature heatmaps (tiled, shared scale)
  * surface           - lowest-layer tilted flat color plane (color = temperature)
  * cross_section     - stack cross-section schematic (memory | thermal-Si | memory)
  * table             - layer-properties table (thickness, in/cross-plane T.C.)
  * future            - "future 3D integration" block diagram
  * peak_per_tier     - peak temperature up the stack (hybrid vs pure DRAM)

The driver (run_experiment.py) calls `make_plots()`; the individual functions
are importable on their own too.
"""

import configparser
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless / no display
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

KELVIN = 273.15
CMAP = "inferno"                 # per-layer heatmaps
SURFACE_CMAP = "jet"             # flat color-plane (rainbow, blue=cold -> red=hot)
DPI = 150

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "figure.dpi":   DPI,
    "savefig.dpi":  DPI,
    "savefig.bbox": "tight",
})

# floorplan file -> friendly layer name (covers both experiments)
FLP_NAMES = {
    "gpu_substrate_flp.csv": "GPU substrate/TSV",
    "gpu_feol_flp.csv":      "GPU FEOL (heat source)",
    "gpu_beol_flp.csv":      "GPU BEOL",
    "ubump_flp.csv":         "GPU-HBM uBump",
    "dram_tier_flp.csv":     "DRAM die",
    "nv_tier_flp.csv":       "NVDRAM die",
    "daf_flp.csv":           "DAF bond",
    # detailed sublayers (EXP_DRAM_R + EXP_NVDRAM_S)
    "bspdn_flp.csv":         "BSPDN",
    "beol_mxy_flp.csv":      "BEOL_MXY",
    "oxide_flp.csv":         "Oxide",
    "hbm_base_beol_flp.csv": "HBM base BEOL",
    "hbm_base_si_flp.csv":   "HBM base die",
    "hybrid_bond_flp.csv":   "Hybrid bonding",
    "dram_beol_flp.csv":     "DRAM die BEOL",
    "dram_die_beol_flp.csv": "DRAM die BEOL",
    "nv_die_beol_flp.csv":   "NVDRAM die BEOL",
    "tim_flp.csv":           "TIM",
    "lid_flp.csv":           "Lid",
}

# material label -> friendly name + colour (for the cross-section / table)
SURROUND = {"THERMAL_SI", "FILLER"}
MAT_NAME = {
    "GPU_Si": "GPU substrate (Si)", "GPU_FEOL": "GPU FEOL", "GPU_BEOL": "GPU BEOL",
    "UBUMP": "GPU-mem uBump", "NV_FEOL": "NVDRAM die", "DRAM_Si": "DRAM die",
    "THERMAL_SI": "Thermal silicon", "DAF": "Die-attach film", "FILLER": "Filler",
    # detailed EXP_DRAM_R materials
    "BSPDN": "BSPDN", "BEOL_MXY": "BEOL_MXY", "OXIDE": "Oxide",
    "GPU_HBM_UBUMP": "GPU-HBM uBump", "HBM_BASE_SI": "HBM base die",
    "HBM_BASE_BEOL": "HBM base BEOL", "HYBRID_BOND": "Hybrid bonding",
    "DRAM_BEOL": "DRAM die BEOL", "DRAM_SI": "DRAM die", "TIM": "TIM", "LID": "Lid",
    "NV_DIE_SI": "NVDRAM die", "NV_DIE_BEOL": "NVDRAM die BEOL",
}
MAT_COLOR = {
    "GPU_Si": "#9ecae1", "GPU_FEOL": "#fb6a4a", "GPU_BEOL": "#fcbba1",
    "UBUMP": "#dadaeb", "NV_FEOL": "#74c476", "DRAM_Si": "#6baed6",
    "THERMAL_SI": "#3182bd", "DAF": "#fdd0a2", "FILLER": "#d9d9d9",
    # detailed EXP_DRAM_R materials
    "BSPDN": "#fdae6b", "BEOL_MXY": "#fee391", "OXIDE": "#f0f0f0",
    "GPU_HBM_UBUMP": "#dadaeb", "HBM_BASE_SI": "#9ecae1",
    "HBM_BASE_BEOL": "#fdd0a2", "HYBRID_BOND": "#c7e9c0",
    "DRAM_BEOL": "#fdd0a2", "DRAM_SI": "#6baed6", "TIM": "#fa9fb5", "LID": "#bdbdbd",
    "NV_DIE_SI": "#74c476", "NV_DIE_BEOL": "#c7e9c0",
}
DEFAULT_COLOR = "#cccccc"


# ── shared IO ────────────────────────────────────────────────────────────────
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


def read_lcf(lcf_path):
    """Return [(layer_id, floorplan_file, thickness_m), ...] in stack order."""
    rows = []
    with open(lcf_path) as fh:
        next(fh, None)
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) >= 3:
                rows.append((int(parts[0]), parts[1], float(parts[2])))
    return rows


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


def layer_labels(lcf_path, n_device):
    """{layer_id: title}; the single trailing layer PACT appends is the lid."""
    labels = {}
    if lcf_path and os.path.exists(lcf_path):
        for idx, flp, _ in read_lcf(lcf_path):
            labels[idx] = FLP_NAMES.get(flp, flp)
    labels[n_device] = "Cooling lid (NoPackage)"
    return labels


def layer_materials(flp_path):
    """Classify a layer floorplan -> (kind, macro_label, surround_label)."""
    uniq = []
    with open(flp_path) as fh:
        next(fh, None)
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) >= 7 and parts[6] not in uniq:
                uniq.append(parts[6])
    if len(uniq) == 1:
        return "full", uniq[0], None
    surround = [l for l in uniq if l in SURROUND]
    macro = [l for l in uniq if l not in SURROUND]
    return "tier", (macro[0] if macro else uniq[0]), (surround[0] if surround else None)


# ── heatmap overlay geometry ─────────────────────────────────────────────────
def _mem_extents(side, gpu_side, mem_side):
    gap = gpu_side - 2 * mem_side
    margin, central = gap / 4.0, gap / 2.0
    a = [margin, margin + mem_side, margin + mem_side + central,
         margin + 2 * mem_side + central]
    f = [int(round(x / gpu_side * side)) for x in a]
    return (f[0], f[1]), (f[2], f[3])


def _draw_mem_boxes(ax, side, gpu_side, mem_side):
    (lo1, hi1), (lo2, hi2) = _mem_extents(side, gpu_side, mem_side)
    for (x0, x1) in ((lo1, hi1), (lo2, hi2)):
        for (y0, y1) in ((lo1, hi1), (lo2, hi2)):
            ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="cyan", lw=1.0, ls="--"))


def under_stack_range(g, gpu_side, mem_side):
    """(min, max) temperature over the die area UNDER the 2×2 memory stacks,
    excluding the cross/frame 'surround' cells.

    In the mold-filled baseline (EXP_*_B) that surround is a near-insulating
    column that traps GPU heat and cooks to a few-hundred °C — an artefact of the
    simplified full-die-uniform-power geometry, not the real heat path. Under a
    stack is where heat actually escapes, so that peak is the meaningful GPU-
    substrate temperature (and, for a memory tier, the real die temperature).
    """
    (lo1, hi1), (lo2, hi2) = _mem_extents(g.shape[0], gpu_side, mem_side)
    m = np.zeros(g.shape, dtype=bool)
    for (x0, x1) in ((lo1, hi1), (lo2, hi2)):
        for (y0, y1) in ((lo1, hi1), (lo2, hi2)):
            m[y0:y1, x0:x1] = True
    if not m.any():
        return float(np.nanmin(g)), float(np.nanmax(g))
    vals = g[m]
    return float(vals.min()), float(vals.max())


def _layer_range(g, gpu_side, mem_side, mask_surround):
    """(vmin, vmax) for one layer: under-stack range when masking, else full grid."""
    if mask_surround:
        return under_stack_range(g, gpu_side, mem_side)
    return float(g.min()), float(g.max())


# ── plots ────────────────────────────────────────────────────────────────────
def heatmaps(prefix, lcf, out, gpu_side, mem_side, include_package=False,
             mask_surround=False):
    all_ids = discover_layers(prefix)
    if not all_ids:
        raise SystemExit(f"No '{prefix}.layer*' files. Run PACT first.")
    n_device = max(all_ids)
    labels = layer_labels(lcf, n_device)
    ids = all_ids if include_package else [i for i in all_ids if i < n_device]
    grids = {i: load_grid(prefix, i) for i in ids}
    grids = {i: g for i, g in grids.items() if g is not None}
    ranges = {i: _layer_range(g, gpu_side, mem_side, mask_surround)
              for i, g in grids.items()}
    vmin = min(r[0] for r in ranges.values())
    vmax = max(r[1] for r in ranges.values())
    ids = sorted(grids)
    ncols = min(4, len(ids))
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False)
    im = None
    for ax, lid in zip(axes.flat, ids):
        g = grids[lid]
        im = ax.imshow(g, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
        _draw_mem_boxes(ax, g.shape[0], gpu_side, mem_side)
        peak = ranges[lid][1]
        ax.set_title(f"L{lid}: {labels.get(lid, '?')}\nmax {peak:.1f} °C", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    surround_note = ("surround = mold; peaks reported under the stacks (mold gap off-scale)"
                     if mask_surround else "surround = thermal silicon")
    fig.suptitle(f"Steady-state temperature per layer\n"
                 f"(dashed boxes = memory stacks; {surround_note})", fontsize=13)
    fig.colorbar(im, ax=axes, label="Temperature (°C)", fraction=0.015, pad=0.02)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def surface_plane(prefix, out, gpu_side, lcf=None, layer_id=None,
                  mem_side=0.010, mask_surround=False):
    """Lowest layer as a tilted flat color plane (color = temperature, no height)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ids = discover_layers(prefix)
    if not ids:
        raise SystemExit(f"No '{prefix}.layer*' files. Run PACT first.")
    lid = layer_id if layer_id is not None else min(ids)
    label = layer_labels(lcf, max(ids)).get(lid, f"layer {lid}") if lcf else f"layer {lid}"
    g = load_grid(prefix, lid)
    side = g.shape[0]
    ext = gpu_side * 1e3
    xs = np.linspace(0, ext, side); ys = np.linspace(0, ext, side)
    X, Y = np.meshgrid(xs, ys)
    cmap = plt.get_cmap(SURFACE_CMAP)
    vmn, vmx = _layer_range(g, gpu_side, mem_side, mask_surround)
    norm = Normalize(vmin=vmn, vmax=vmx, clip=mask_surround)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=cmap(norm(g)), shade=False,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.view_init(elev=35, azim=-60)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.05))
    ax.set_title(f"Layer {lid}: {label} — 2D temperature map")
    peak_note = "\n(under stacks; mold gap off-scale)" if mask_surround else ""
    ax.text2D(0.02, 0.95, f"Peak: {vmx:.1f} °C{peak_note}", transform=ax.transAxes,
              fontsize=13, fontweight="bold", color="#b30000",
              bbox=dict(boxstyle="round", fc="white", ec="#b30000", alpha=0.85))
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array(g)
    fig.colorbar(m, ax=ax, label="Temperature (°C)", fraction=0.03, pad=0.10, shrink=0.6)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def _flat_plane(ax, g, gpu_side, title, peak_color="#b30000",
                mem_side=0.010, mask_surround=False):
    """Render one layer as a tilted flat color plane (paper-style); returns the
    ScalarMappable so the caller can attach a per-panel colorbar."""
    side = g.shape[0]
    ext = gpu_side * 1e3                      # mm
    xs = np.linspace(0, ext, side); ys = np.linspace(0, ext, side)
    X, Y = np.meshgrid(xs, ys)
    cmap = plt.get_cmap(SURFACE_CMAP)
    vmn, vmx = _layer_range(g, gpu_side, mem_side, mask_surround)
    norm = Normalize(vmin=vmn, vmax=vmx, clip=mask_surround)
    ax.plot_surface(X, Y, np.zeros_like(g), facecolors=cmap(norm(g)), shade=False,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.view_init(elev=35, azim=-60)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.05))
    ax.set_title(title, fontsize=12, pad=0)
    peak_note = "  (under stacks)" if mask_surround else ""
    ax.text2D(0.5, -0.02, f"Peak Temperature{peak_note}\n{vmx:.1f} °C", transform=ax.transAxes,
              ha="center", va="top", fontsize=12, fontweight="bold", color=peak_color,
              bbox=dict(boxstyle="round", fc="white", ec=peak_color, alpha=0.9))
    ax.text2D(0.5, 0.0, f"← {ext:g} mm →", transform=ax.transAxes,
              ha="center", va="bottom", fontsize=8, color="#444444")
    m = plt.cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array(g)
    return m


def side_by_side(prefix, out, gpu_side, lcf=None,
                 left=(1, "GPU substrate (compute die)"),
                 right=(9, "Lowest NVDRAM tier"),
                 mem_side=0.010, mask_surround=False):
    """Paper-style two-panel figure: two layers shown as tilted flat color planes,
    each with its own jet colour scale and a peak-temperature callout."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    ids = discover_layers(prefix)
    if not ids:
        raise SystemExit(f"No '{prefix}.layer*' files. Run PACT first.")
    fig = plt.figure(figsize=(13, 5.6))
    for pos, (lid, label) in enumerate((left, right), start=1):
        g = load_grid(prefix, lid)
        if g is None:
            raise SystemExit(f"side_by_side: missing grid for layer {lid}.")
        ax = fig.add_subplot(1, 2, pos, projection="3d")
        m = _flat_plane(ax, g, gpu_side, f"L{lid}: {label}",
                        mem_side=mem_side, mask_surround=mask_surround)
        fig.colorbar(m, ax=ax, label="Temperature (°C)", fraction=0.03, pad=0.10, shrink=0.6)
    fig.suptitle("3D-stacked GPU + NVDRAM — steady-state temperature\n"
                 "(independent colour scales, as in the IEDM figure)", fontsize=13)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def cross_section(lcf, config, flp_dir, out):
    layers = read_lcf(lcf)
    ks = material_k(config)
    n = len(layers)
    th = np.array([t for _, _, t in layers])
    h = np.log10(th / th.min()) + 1.0
    h = h / h.sum()
    fig, ax = plt.subplots(figsize=(8.5, max(7, 0.42 * n)))
    y, used = 0.0, {}
    for (lid, flp, t), height in zip(layers, h):
        kind, macro, surround = layer_materials(os.path.join(flp_dir, flp))
        if kind == "full":
            col = MAT_COLOR.get(macro, DEFAULT_COLOR)
            ax.add_patch(plt.Rectangle((0.05, y), 0.90, height, fc=col, ec="black", lw=0.6))
            used[macro] = col
        else:
            mcol = MAT_COLOR.get(macro, DEFAULT_COLOR)
            scol = MAT_COLOR.get(surround, DEFAULT_COLOR)
            ax.add_patch(plt.Rectangle((0.05, y), 0.34, height, fc=mcol, ec="black", lw=0.6))
            ax.add_patch(plt.Rectangle((0.39, y), 0.17, height, fc=scol, ec="black", lw=0.6))
            ax.add_patch(plt.Rectangle((0.56, y), 0.34, height, fc=mcol, ec="black", lw=0.6))
            used[macro] = mcol
            if surround:
                used[surround] = scol
        ax.text(0.965, y + height / 2.0,
                f"L{lid} {MAT_NAME.get(macro, macro)}  ({t*1e6:g} µm)",
                va="center", ha="left", fontsize=7.5)
        y += height
    ax.add_patch(FancyBboxPatch((0.05, y + 0.005), 0.85, 0.03, boxstyle="round,pad=0.002",
                                fc="#bdd7e7", ec="black"))
    ax.text(0.475, y + 0.02, "Liquid cold-plate lid (top cooling)", ha="center",
            va="center", fontsize=8, style="italic")
    ax.add_patch(FancyBboxPatch((0.05, -0.035), 0.85, 0.03, boxstyle="round,pad=0.002",
                                fc="#d9d9d9", ec="black"))
    ax.text(0.475, -0.02, "Laminate / package (adiabatic bottom)", ha="center",
            va="center", fontsize=8, style="italic")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black") for c in used.values()]
    txt = [f"{MAT_NAME.get(l, l)}  (k={ks.get(l, float('nan')):.0f} W/m·K)" for l in used]
    ax.legend(handles, txt, loc="center left", bbox_to_anchor=(-0.55, 0.5),
              fontsize=7.5, frameon=True)
    ax.set_xlim(-0.05, 1.35); ax.set_ylim(-0.06, y + 0.06); ax.axis("off")
    ax.set_title("Cross-section stack (bottom→top)\n"
                 "memory tiers: memory | thermal-silicon | memory", fontsize=12)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def layer_table(lcf, config, flp_dir, out):
    layers = read_lcf(lcf)
    ks = material_k(config)
    rows = []
    for lid, flp, t in layers:
        _, macro, _ = layer_materials(os.path.join(flp_dir, flp))
        name = MAT_NAME.get(macro, macro)
        k = ks.get(macro, float("nan"))
        if rows and rows[-1][0] == name and abs(rows[-1][1] - t) < 1e-15:
            rows[-1][3] += 1
        else:
            rows.append([name, t, k, 1])
    cell, total = [], 0.0
    for name, t, k, cnt in rows:
        total += t * cnt * 1e6
        cell.append([name + (f"  ×{cnt}" if cnt > 1 else ""),
                     f"{t*1e6:g}", f"{k:.1f}", f"{k:.1f}"])
    cell.append(["TOTAL device stack", f"{total:g}", "—", "—"])
    fig, ax = plt.subplots(figsize=(8.5, 0.5 + 0.34 * (len(cell) + 1)))
    ax.axis("off")
    tbl = ax.table(cellText=cell,
                   colLabels=["Stack layer (bottom→top)", "Thickness (µm)",
                              "T.C. in-plane (W/m·K)", "T.C. cross-plane (W/m·K)"],
                   colWidths=[0.34, 0.20, 0.23, 0.23], cellLoc="center",
                   loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for (r, c), cellobj in tbl.get_celld().items():
        if r == 0:
            cellobj.set_facecolor("#3182bd"); cellobj.set_text_props(color="white", fontweight="bold")
        elif r == len(cell):
            cellobj.set_facecolor("#deebf7"); cellobj.set_text_props(fontweight="bold")
    ax.set_title("Cross-section stack layers — thickness & thermal conductivity",
                 fontsize=12, pad=12)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def future_integration(lcf, flp_dir, out):
    layers = read_lcf(lcf)
    nv = sum(1 for _, flp, _ in layers if layer_materials(os.path.join(flp_dir, flp))[1] == "NV_FEOL")
    dram = sum(1 for _, flp, _ in layers if layer_materials(os.path.join(flp_dir, flp))[1] == "DRAM_Si")
    blocks = [
        ("Package / Laminate", "#d9d9d9"),
        ("GPU compute die (heat source)", "#fb6a4a"),
        ("uBump / hybrid-bond interface", "#dadaeb"),
    ]
    if nv:
        blocks.append((f"NVDRAM stack — {nv} tiers (non-volatile, bottom)", "#74c476"))
    blocks.append((f"DRAM stack — {dram} tiers (HBM, top)", "#6baed6"))
    blocks.append(("Liquid cold-plate lid (cooling)", "#bdd7e7"))
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    y, bw, bh, gap = 0.0, 0.8, 0.10, 0.045
    for i, (label, col) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((0.1, y), bw, bh, boxstyle="round,pad=0.008",
                                    fc=col, ec="black", lw=1.0))
        ax.text(0.5, y + bh / 2.0, label, ha="center", va="center", fontsize=10, fontweight="bold")
        if i < len(blocks) - 1:
            ax.annotate("", xy=(0.5, y + bh + gap), xytext=(0.5, y + bh),
                        arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.6))
        y += bh + gap
    ax.text(0.5, -0.04, "heat flows UP to the lid →", ha="center", fontsize=9,
            style="italic", color="#b30000")
    ax.set_xlim(0, 1); ax.set_ylim(-0.07, y + 0.02); ax.axis("off")
    title = ("Future 3D integration: NVDRAM (bottom) + DRAM (top) on GPU" if nv
             else "3D integration: DRAM (HBM) on GPU")
    ax.set_title(title, fontsize=12)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


def _stack_profile(exp_dir, prefix, lcf, flp_regions,
                   gpu_side=0.024, mem_side=0.010, mask_surround=False):
    out = []
    for layer_id, flp, _ in read_lcf(os.path.join(exp_dir, lcf)):
        if flp in flp_regions:
            g = load_grid(os.path.join(exp_dir, prefix), layer_id)
            if g is None:
                raise SystemExit(f"Missing grid for layer {layer_id} in {exp_dir}.")
            peak = (under_stack_range(g, gpu_side, mem_side)[1] if mask_surround
                    else g.max())
            out.append((flp_regions[flp], peak))
    return out


def peak_per_tier(hybrid_dir, hybrid_prefix, hybrid_lcf,
                  dram_dir, dram_prefix, dram_lcf, out,
                  gpu_side=0.024, mem_side=0.010, mask_surround=False):
    """Peak temperature up the stack: NVDRAM+DRAM hybrid (continuous) vs pure DRAM."""
    hybrid = _stack_profile(hybrid_dir, hybrid_prefix, hybrid_lcf,
                            {"nv_tier_flp.csv": "NVDRAM", "dram_tier_flp.csv": "DRAM"},
                            gpu_side, mem_side, mask_surround)
    pure = _stack_profile(dram_dir, dram_prefix, dram_lcf, {"dram_tier_flp.csv": "DRAM"},
                          gpu_side, mem_side, mask_surround)
    if not hybrid or not pure:
        raise SystemExit("peak_per_tier: missing tier data (run both experiments first).")
    regions = [r for r, _ in hybrid]
    yh = np.array([v for _, v in hybrid]); xh = np.arange(1, len(yh) + 1)
    n_nv = sum(1 for r in regions if r == "NVDRAM")
    yp = np.array([v for _, v in pure]); xp = np.arange(1, len(yp) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(xh, yh, "-", color="#7b3294", lw=2.2, zorder=2, label="NVDRAM + DRAM hybrid stack")
    nv_i = [i for i, r in enumerate(regions) if r == "NVDRAM"]
    dr_i = [i for i, r in enumerate(regions) if r == "DRAM"]
    ax.plot(xh[nv_i], yh[nv_i], "o", color="#d62728", ms=8, zorder=3, label="   ↳ NVDRAM tier (bottom)")
    ax.plot(xh[dr_i], yh[dr_i], "s", color="#e6862e", ms=7, zorder=3, label="   ↳ DRAM tier (continues on top)")
    if n_nv:
        ax.axvspan(0.5, n_nv + 0.5, color="#d62728", alpha=0.06, zorder=0)
        ax.axvline(n_nv + 0.5, color="#888888", ls=":", lw=1.2, zorder=1)
        ylo, yhi = ax.get_ylim()
        ax.text(n_nv / 2.0 + 0.5, ylo + 0.04 * (yhi - ylo), "NVDRAM (bottom)",
                ha="center", va="bottom", fontsize=8, color="#d62728")
    ax.plot(xp, yp, "s--", color="#1f77b4", lw=2, ms=6, zorder=2, label="Pure DRAM stack (HBM only)")
    ax.annotate(f"{yh[0]:.1f} °C", (xh[0], yh[0]), textcoords="offset points",
                xytext=(4, 6), color="#7b3294", fontsize=9, fontweight="bold")
    ax.annotate(f"{yp[0]:.1f} °C", (xp[0], yp[0]), textcoords="offset points",
                xytext=(4, -14), color="#1f77b4", fontsize=9, fontweight="bold")
    ax.set_xlabel("Memory tier index (1 = bottom, nearest GPU → top = lid)")
    ax.set_ylabel("Peak die temperature (°C)")
    ax.set_title("Peak temperature up the memory stack\n"
                 "NVDRAM+DRAM hybrid vs pure DRAM", fontsize=12)
    ax.set_xticks(np.arange(1, max(len(yh), len(yp)) + 1))
    ax.grid(True, ls=":", alpha=0.5); ax.legend(fontsize=9)
    fig.savefig(out); plt.close(fig)
    print(f"    Written: {out}")


# ── dispatcher used by the driver ────────────────────────────────────────────
def make_plots(exp_dir, meta, out_dir):
    """Render every figure listed in meta['plots'] into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(exp_dir, meta["grid_prefix"])
    lcf = os.path.join(exp_dir, meta["lcf"])
    config = os.path.join(exp_dir, meta["config"])
    gpu_side = float(meta.get("gpu_side", 0.024))
    mem_side = float(meta.get("mem_side", 0.010))
    # Baseline (mold-surround) experiments set mask_surround=true so every peak is
    # read UNDER the stacks, ignoring the unphysical mold-gap hotspot.
    mask = bool(meta.get("mask_surround", False))
    _report_substrate_peak(prefix, lcf, gpu_side, mem_side, mask)
    for plot in meta.get("plots", []):
        if plot == "heatmaps":
            heatmaps(prefix, lcf, os.path.join(out_dir, "heatmaps.png"), gpu_side, mem_side,
                     mask_surround=mask)
        elif plot == "surface":
            surface_plane(prefix, os.path.join(out_dir, "lowest_layer_surface.png"), gpu_side, lcf,
                          mem_side=mem_side, mask_surround=mask)
        elif plot == "side_by_side":
            sb = meta.get("side_by_side", {})
            left = tuple(sb.get("left", [1, "GPU substrate (compute die)"]))
            right = tuple(sb.get("right", [9, "Lowest NVDRAM tier"]))
            side_by_side(prefix, os.path.join(out_dir, "gpu_vs_nvdram_side_by_side.png"),
                         gpu_side, lcf, left=left, right=right,
                         mem_side=mem_side, mask_surround=mask)
        elif plot == "cross_section":
            cross_section(lcf, config, exp_dir, os.path.join(out_dir, "cross_section.png"))
        elif plot == "table":
            layer_table(lcf, config, exp_dir, os.path.join(out_dir, "layer_table.png"))
        elif plot == "future":
            future_integration(lcf, exp_dir, os.path.join(out_dir, "future_integration.png"))
        elif plot == "peak_per_tier":
            cmp = meta["compare_dram"]
            dram_dir = os.path.join(exp_dir, cmp["dir"])
            peak_per_tier(exp_dir, meta["grid_prefix"], meta["lcf"],
                          dram_dir, cmp["grid_prefix"], cmp["lcf"],
                          os.path.join(out_dir, "peak_per_tier_hybrid_vs_dram.png"),
                          gpu_side, mem_side, mask)
        else:
            print(f"    (skipping unknown plot '{plot}')")


def _report_substrate_peak(prefix, lcf, gpu_side, mem_side, mask_surround):
    """Print the peak GPU-substrate (FEOL) temperature — under the stacks when
    mask_surround, so the mold-gap artefact is excluded."""
    if not os.path.exists(lcf):
        return
    for lid, flp, _ in read_lcf(lcf):
        if flp == "gpu_feol_flp.csv":
            g = load_grid(prefix, lid)
            if g is None:
                return
            if mask_surround:
                _, hi = under_stack_range(g, gpu_side, mem_side)
                print(f"    GPU substrate peak (L{lid}): {hi:.1f} °C  "
                      f"(under stacks; excludes mold-gap artefact, raw grid max {g.max():.1f} °C)")
            else:
                print(f"    GPU substrate peak (L{lid}): {g.max():.1f} °C")
            return
