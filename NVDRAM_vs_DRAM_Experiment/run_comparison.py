"""
NVDRAM vs 3D-Bonded DRAM — Thermal Comparison
==============================================
Runs PACT for both chip configurations then generates side-by-side
publication figures comparing their steady-state and transient thermal
behaviour.

Usage (from inside NVDRAM_vs_DRAM_Experiment/):
    python run_comparison.py [--skip-sim] [--out-dir figures/]

    --skip-sim   Skip PACT runs; use existing output files (fast re-plot).
    --out-dir    Directory for PNG output (default: figures/).

Figures produced:
    1. heatmap_comparison.png         — steady-state 40x40 grid, NVDRAM vs DRAM, all layers
    2. peak_temperature_comparison.png — peak/mean/min bar chart per layer per chip
    3. block_transient_comparison.png  — per-block temperature evolution for both chips
    4. thermal_resistance_breakdown.png— analytic R_th per layer, side by side
    5. power_budget_comparison.png     — total chip power per scenario for both chips
"""

import argparse
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ── constants ────────────────────────────────────────────────────────────────
KELVIN       = 273.15
DPI          = 150
CMAP         = "inferno"
FONT_TITLE   = 13
FONT_AXIS    = 11
PTRACE_SCEN  = ["Idle", "Active Read", "Active Write"]

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_AXIS,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":     DPI,
    "savefig.dpi":    DPI,
    "savefig.bbox":   "tight",
})

# Physical layer descriptions used in legends / titles
NVDRAM_LAYERS = {
    0: "CMOS (Si, 300 µm)",
    1: "1T1C Mem L1 (PolySi, 2 µm)",
    2: "1T1C Mem L2 (PolySi, 2 µm)",
    3: "Heat Spreader",
    4: "Heat Sink",
}
DRAM_LAYERS = {
    0: "CMOS (Si, 300 µm)",
    1: "Fusion Bond (SiO₂, 2 µm)",
    2: "DRAM Die (Si, 50 µm)",
    3: "Heat Spreader",
    4: "Heat Sink",
}

# Analytic thermal resistance per layer (R = thickness / (k * area))
# area = 8.5mm * 8.5mm = 72.25e-6 m²
CHIP_AREA = 0.0085 * 0.0085

LAYER_PROPS = {
    "nvdram": [
        ("CMOS",           0.0003,   130.0),
        ("1T1C Mem L1",    0.000002,  14.0),
        ("1T1C Mem L2",    0.000002,  14.0),
    ],
    "dram": [
        ("CMOS",           0.0003,   130.0),
        ("Fusion Bond",    0.000002,   1.4),
        ("DRAM Die",       0.00005,  130.0),
    ],
}


def k2c(v):
    return np.asarray(v, dtype=float) - KELVIN


# ── PACT runner ──────────────────────────────────────────────────────────────

def run_pact(exp_dir, lcf, config, params, prefix):
    pact_script = os.path.join(exp_dir, "..", "src", "PACT.py")
    pact_script = os.path.realpath(pact_script)
    cmd = [
        sys.executable, pact_script,
        lcf, config, params,
        "--gridSteadyFile", prefix,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=exp_dir, capture_output=False)
    if result.returncode != 0:
        print(f"  [WARN] PACT exited with code {result.returncode} for {prefix}")


# ── data loaders ─────────────────────────────────────────────────────────────

def load_grid_steady(exp_dir, prefix, layer_id):
    path = os.path.join(exp_dir, f"{prefix}.grid.steady.layer{layer_id}")
    if not os.path.exists(path):
        return None
    return k2c(np.loadtxt(path)).reshape(40, 40)


def load_block_transient(exp_dir, prefix, step_size_s=0.00333):
    path = os.path.join(exp_dir, f"{prefix}.block.transient.csv")
    if not os.path.exists(path):
        return {}
    records = []
    current_step, current_layer = None, None
    step_re  = re.compile(r"^step\s+(\d+)\s+layer number:(\d+)")
    layer_re = re.compile(r"^layer number:(\d+)")
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            m = step_re.match(line)
            if m:
                current_step, current_layer = int(m.group(1)), int(m.group(2))
                continue
            m = layer_re.match(line)
            if m:
                current_layer = int(m.group(1))
                continue
            if current_step is None:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    records.append((current_step, current_layer, parts[1], float(parts[2])))
                except (ValueError, IndexError):
                    pass
    if not records:
        return {}
    df = pd.DataFrame(records, columns=["step", "layer", "block", "temp_C"])
    return {lid: grp.reset_index(drop=True) for lid, grp in df.groupby("layer")}


# ── figure helpers ────────────────────────────────────────────────────────────

def _colorbar(fig, im, ax, label):
    fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)


# ── figure 1: heatmap comparison ─────────────────────────────────────────────

def fig_heatmap_comparison(exp_dir, out_dir):
    configs = [
        ("nvdram", NVDRAM_LAYERS, "NVDRAM (Monolithic 3D)"),
        ("dram",   DRAM_LAYERS,   "3D-Bonded DRAM"),
    ]
    active_layers = [0, 1, 2]

    grids = {}
    for tag, lnames, _ in configs:
        grids[tag] = {lid: load_grid_steady(exp_dir, tag, lid) for lid in active_layers}

    # global colour range
    all_vals = [g for tag in grids for lid, g in grids[tag].items() if g is not None]
    if not all_vals:
        print("  [skip] no steady-state grid files found")
        return
    vmin = min(g.min() for g in all_vals)
    vmax = max(g.max() for g in all_vals)

    fig, axes = plt.subplots(2, len(active_layers), figsize=(5 * len(active_layers), 9))
    fig.suptitle("Steady-State Thermal Maps: NVDRAM vs 3D-Bonded DRAM",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for row_idx, (tag, lnames, chip_title) in enumerate(configs):
        for col_idx, lid in enumerate(active_layers):
            ax  = axes[row_idx][col_idx]
            g   = grids[tag].get(lid)
            lbl = lnames.get(lid, f"Layer {lid}")
            if g is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(g, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
            if row_idx == 0:
                ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Column")
            ax.set_ylabel(chip_title if col_idx == 0 else "")
            ax.set_yticks([])
            ax.set_xticks([])

    fig.colorbar(im, ax=axes, label="Temperature (°C)", fraction=0.015, pad=0.02)
    out = os.path.join(out_dir, "heatmap_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 2: peak temperature comparison ────────────────────────────────────

def fig_peak_temp_comparison(exp_dir, out_dir):
    configs = [
        ("nvdram", NVDRAM_LAYERS, "NVDRAM",  "#d62728"),
        ("dram",   DRAM_LAYERS,   "3D DRAM", "#1f77b4"),
    ]
    active_layers = [0, 1, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Layer-wise Temperature Statistics: NVDRAM vs 3D-Bonded DRAM",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    metrics = [("peak", "Peak"),  ("mean", "Mean"),  ("min", "Min")]

    for ax, (metric, mlabel) in zip(axes, metrics):
        x        = np.arange(len(active_layers))
        width    = 0.35
        for offset, (tag, lnames, chip_label, color) in enumerate(configs):
            vals = []
            for lid in active_layers:
                g = load_grid_steady(exp_dir, tag, lid)
                if g is None:
                    vals.append(0.0)
                elif metric == "peak":
                    vals.append(g.max())
                elif metric == "mean":
                    vals.append(g.mean())
                else:
                    vals.append(g.min())
            xpos = x + (offset - 0.5) * width
            bars = ax.bar(xpos, vals, width, label=chip_label, color=color, alpha=0.82)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([NVDRAM_LAYERS[lid].split("(")[0].strip()
                            for lid in active_layers], rotation=15, ha="right")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"{mlabel} Temperature per Layer")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(out_dir, "peak_temperature_comparison.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 3: block transient comparison ─────────────────────────────────────

def fig_block_transient_comparison(exp_dir, out_dir):
    configs = [
        ("nvdram", NVDRAM_LAYERS, "NVDRAM"),
        ("dram",   DRAM_LAYERS,   "3D-Bonded DRAM"),
    ]
    active_layers = [0, 1, 2]
    n_row = len(configs)
    n_col = len(active_layers)

    fig, axes = plt.subplots(n_row, n_col, figsize=(6 * n_col, 4.5 * n_row))
    fig.suptitle("Block Temperature Transient: NVDRAM vs 3D-Bonded DRAM",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    for row_idx, (sim_tag, lnames, chip_label) in enumerate(configs):
        layer_data = load_block_transient(exp_dir, sim_tag)
        for col_idx, lid in enumerate(active_layers):
            ax  = axes[row_idx][col_idx]
            df  = layer_data.get(lid)
            lbl = lnames.get(lid, f"Layer {lid}")
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{chip_label} — {lbl}")
                continue
            steps = sorted(df["step"].unique())
            time_s = [(s - 1) * 0.00333 * 100 for s in steps]
            for b_idx, block in enumerate(df["block"].unique()):
                sub   = df[df["block"] == block].sort_values("step")
                temps = sub["temp_C"].values
                n_pts = min(len(time_s), len(temps))
                ax.plot(time_s[:n_pts], temps[:n_pts],
                        label=block, color=palette[b_idx], linewidth=1.8)
            ax.set_title(f"{chip_label} — {lbl}", fontsize=10)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Temperature (°C)")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

    out = os.path.join(out_dir, "block_transient_comparison.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 4: analytic thermal resistance breakdown ───────────────────────────

def fig_thermal_resistance(out_dir):
    fig, ax = plt.subplots(figsize=(11, 6))

    chips   = list(LAYER_PROPS.keys())
    colors  = {"nvdram": "#d62728", "dram": "#1f77b4"}
    labels  = {"nvdram": "NVDRAM (Monolithic 3D)", "dram": "3D-Bonded DRAM"}
    x_all   = []
    bar_kw  = dict(width=0.35, alpha=0.85)

    x = np.arange(3)   # three layers per chip
    for offset, chip in enumerate(chips):
        layers = LAYER_PROPS[chip]
        r_vals = [thick / (k * CHIP_AREA) * 1e3
                  for name, thick, k in layers]   # mK/W
        xpos = x + (offset - 0.5) * 0.35
        bars = ax.bar(xpos, r_vals, label=labels[chip], color=colors[chip], **bar_kw)
        for bar, (name, thick, k), rv in zip(bars, layers, r_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, rv + 0.01,
                    f"{rv:.2f}", ha="center", va="bottom", fontsize=8)

    layer_names = [f"Layer {i}" for i in range(3)]
    nvdram_names = [n for n, _, _ in LAYER_PROPS["nvdram"]]
    dram_names   = [n for n, _, _ in LAYER_PROPS["dram"]]
    xtick_labels = [f"{a}\n(NVDRAM)\n{b}\n(DRAM)"
                    for a, b in zip(nvdram_names, dram_names)]

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_ylabel("Thermal Resistance R_th (mK/W)")
    ax.set_title("Analytic Layer Thermal Resistance: NVDRAM vs 3D-Bonded DRAM\n"
                 r"$R_{th} = \mathrm{thickness} / (k \cdot A_{chip})$",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # annotate key insight
    ax.annotate(
        "FusionBond (k=1.4 W/m·K)\ncreates 10× higher R_th\nthan PolySi1T1C (k=14 W/m·K)\nat same thickness",
        xy=(1.175, ax.get_ylim()[1] * 0.6),
        xytext=(1.5, ax.get_ylim()[1] * 0.75),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange"),
    )

    out = os.path.join(out_dir, "thermal_resistance_breakdown.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 5: total power budget comparison ───────────────────────────────────

def fig_power_budget(exp_dir, out_dir):
    power_files = {
        "nvdram": ["nvdram_cmos_ptrace.csv",
                   "nvdram_mem_ptrace.csv",
                   "nvdram_mem_ptrace.csv"],
        "dram":   ["dram_cmos_ptrace.csv",
                   "dram_bond_ptrace.csv",
                   "dram_die_ptrace.csv"],
    }
    layer_labels = {
        "nvdram": ["CMOS", "1T1C Mem L1", "1T1C Mem L2"],
        "dram":   ["CMOS", "Fusion Bond", "DRAM Die"],
    }
    chip_colors = {
        "nvdram": ["#d62728", "#ff7f0e", "#e377c2"],
        "dram":   ["#1f77b4", "#aec7e8", "#17becf"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Total Power Budget per Scenario: NVDRAM vs 3D-Bonded DRAM",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for ax, scen_idx, scen_name in zip(axes, range(3), PTRACE_SCEN):
        bottom_nvdram = np.zeros(1)
        bottom_dram   = np.zeros(1)
        for chip, col_prefix in [("nvdram", 0), ("dram", 1)]:
            bottom = np.zeros(1)
            x_pos  = 0 if chip == "nvdram" else 1
            for layer_idx, ptrace_file in enumerate(power_files[chip]):
                path = os.path.join(exp_dir, ptrace_file)
                df   = pd.read_csv(path, skipinitialspace=True)
                df.columns = [c.strip() for c in df.columns]
                col = df.columns[scen_idx + 1]  # Power, Power1, Power2
                total = df[col].sum()
                ax.bar(x_pos, total, bottom=bottom,
                       color=chip_colors[chip][layer_idx],
                       label=f"{chip.upper()} {layer_labels[chip][layer_idx]}",
                       alpha=0.85, width=0.5)
                ax.text(x_pos, float(bottom.item()) + total / 2,
                        f"{total:.2f}W", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
                bottom += total

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NVDRAM", "3D DRAM"], fontsize=11)
        ax.set_ylabel("Power (W)")
        ax.set_title(f"Scenario: {scen_name}")
        ax.grid(True, axis="y", alpha=0.3)
        if scen_idx == 0:
            handles = [
                mpatches.Patch(color=chip_colors["nvdram"][i],
                               label=f"NVDRAM: {layer_labels['nvdram'][i]}")
                for i in range(3)
            ] + [
                mpatches.Patch(color=chip_colors["dram"][i],
                               label=f"DRAM: {layer_labels['dram'][i]}")
                for i in range(3)
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper left")

    out = os.path.join(out_dir, "power_budget_comparison.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 6: thermal resistance scaling (analytic, N = 1..n_max) ─────────────

def fig_thermal_scaling(out_dir, n_max=8):
    """Analytic R_th vs N for NVDRAM-only, DRAM-only, and hybrid stacks."""
    CHIP_AREA = 0.0085 * 0.0085

    def r(thickness, k):
        return thickness / (k * CHIP_AREA) * 1e3   # mK/W

    R_CMOS     = r(3.0e-4, 130.0)
    R_1T1C     = r(2.0e-6,  14.0)
    R_BOND     = r(2.0e-6,   1.4)
    R_DRAM_DIE = r(5.0e-5, 130.0)

    ns = list(range(1, n_max + 1))

    r_nvdram = [R_CMOS + n * R_1T1C                          for n in ns]
    r_dram   = [R_CMOS + n * (R_BOND + R_DRAM_DIE)          for n in ns]
    r_hybrid = [R_CMOS + n * R_1T1C + n * (R_BOND + R_DRAM_DIE) for n in ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns, r_nvdram, "o-",  color="#d62728", lw=2,   label="NVDRAM only (N × 1T1C layers)")
    ax.plot(ns, r_dram,   "s--", color="#1f77b4", lw=2,   label="DRAM only (N × bond+die)")
    ax.plot(ns, r_hybrid, "^-.", color="#2ca02c", lw=2,   label="Hybrid (N NVDRAM + N DRAM)")

    # annotate slope labels at N=4
    mid = 4 - 1
    for vals, label, color, dy in [
        (r_nvdram, f"+{R_1T1C:.1f} mK/W per layer",          "#d62728",  4),
        (r_dram,   f"+{R_BOND+R_DRAM_DIE:.1f} mK/W per die", "#1f77b4", -8),
        (r_hybrid, f"+{R_1T1C+R_BOND+R_DRAM_DIE:.1f} mK/W per N", "#2ca02c", 4),
    ]:
        ax.annotate(label, xy=(ns[mid], vals[mid]),
                    xytext=(ns[mid] + 0.3, vals[mid] + dy),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xlabel("N (number of memory layers / dies per type)")
    ax.set_ylabel("Total Stack R_th (mK/W)")
    ax.set_title("Thermal Resistance Scaling vs Number of Stacked Layers",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.set_xticks(ns)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = os.path.join(out_dir, "thermal_scaling.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 7: hybrid stack layer-by-layer temperature profile ─────────────────

def fig_hybrid_layer_profile(exp_dir, out_dir, n):
    """Mean temperature at each layer boundary in the hybrid N-layer stack."""
    prefix = f"hybrid_n{n}"

    # Build expected layer descriptions matching generate_stack.py hybrid layout:
    # layer 0: CMOS, layers 1..n: 1T1C, layers n+1..n+2n: alternating bond/die
    layer_labels = ["CMOS (Si)"]
    for i in range(1, n + 1):
        layer_labels.append(f"1T1C L{i} (PolySi)")
    for i in range(1, n + 1):
        layer_labels.append(f"Bond {i} (SiO₂)")
        layer_labels.append(f"DRAM Die {i} (Si)")
    # PACT appends heatspreader and heatsink
    layer_labels += ["Heat Spreader", "Heat Sink"]

    means, peaks, mins = [], [], []
    found_layers = []
    for lid in range(len(layer_labels)):
        g = load_grid_steady(exp_dir, prefix, lid)
        if g is None:
            break
        found_layers.append(lid)
        means.append(g.mean())
        peaks.append(g.max())
        mins.append(g.min())

    if not found_layers:
        print(f"  [skip] no grid.steady files found for {prefix}")
        return

    labels = layer_labels[:len(found_layers)]
    y = np.arange(len(found_layers))

    fig, ax = plt.subplots(figsize=(7, max(5, len(found_layers) * 0.55)))
    ax.barh(y, peaks,  height=0.6, color="#d62728", alpha=0.35, label="Peak")
    ax.barh(y, means,  height=0.6, color="#1f77b4", alpha=0.75, label="Mean")
    ax.barh(y, mins,   height=0.6, color="#2ca02c", alpha=0.35, label="Min")

    # shade NVDRAM section
    ax.axhspan(0.5, n + 0.5,            facecolor="#ffdddd", alpha=0.25, label="NVDRAM region")
    ax.axhspan(n + 0.5, n * 3 + 0.5,   facecolor="#ddddff", alpha=0.25, label="DRAM region")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_title(f"Hybrid Stack Layer Temperature Profile  (N={n})\n"
                 f"Bottom = package side, top = far from package",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()   # layer 0 at top (closest to package)

    out = os.path.join(out_dir, f"hybrid_n{n}_layer_profile.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 8: hybrid stack heatmaps (all active layers) ──────────────────────

def fig_hybrid_heatmaps(exp_dir, out_dir, n):
    """40×40 heatmaps for all active layers in the hybrid stack."""
    prefix = f"hybrid_n{n}"
    n_active = 1 + n + n * 2   # CMOS + N 1T1C + N (bond+die)

    layer_labels = ["CMOS"]
    for i in range(1, n + 1):
        layer_labels.append(f"1T1C L{i}")
    for i in range(1, n + 1):
        layer_labels.append(f"Bond {i}")
        layer_labels.append(f"DRAM {i}")

    grids, labels = [], []
    for lid in range(n_active):
        g = load_grid_steady(exp_dir, prefix, lid)
        if g is not None:
            grids.append(g)
            labels.append(layer_labels[lid] if lid < len(layer_labels) else f"L{lid}")

    if not grids:
        print(f"  [skip] no grid.steady files for {prefix}")
        return

    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    ncols = min(len(grids), 4)
    nrows = (len(grids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(f"Hybrid Stack Steady-State Thermal Maps  (N={n})",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for idx, (g, lbl) in enumerate(zip(grids, labels)):
        ax = axes[idx // ncols][idx % ncols]
        im = ax.imshow(g, origin="lower", cmap=CMAP, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        # shade title background by region
        color = "#ffcccc" if "1T1C" in lbl else ("#ccccff" if ("Bond" in lbl or "DRAM" in lbl) else "#eeeeee")
        ax.title.set_backgroundcolor(color)

    for idx in range(len(grids), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.colorbar(im, ax=axes, label="Temperature (°C)", fraction=0.015, pad=0.02)
    out = os.path.join(out_dir, f"hybrid_n{n}_heatmaps.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip PACT runs, use existing output files")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--n", type=int, default=4,
                        help="Number of memory layers per type for HBM-style stacks (default: 4)")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(exp_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n = args.n

    print(f"Experiment dir : {exp_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"N              : {n} layers per type\n")

    if not args.skip_sim:
        print("=== Generating LCF files ===")
        import generate_stack
        generate_stack.generate("nvdram", n)
        generate_stack.generate("dram",   n)
        generate_stack.generate("hybrid", n)

        print("\n=== Running PACT simulations ===")
        runs = [
            ("nvdram_lcf.csv",          "nvdram.grid.steady",        "NVDRAM (N=1)"),
            ("dram_lcf.csv",            "dram.grid.steady",          "DRAM (N=1)"),
            (f"nvdram_n{n}_lcf.csv",   f"nvdram_n{n}.grid.steady",  f"NVDRAM (N={n})"),
            (f"dram_n{n}_lcf.csv",     f"dram_n{n}.grid.steady",    f"DRAM (N={n})"),
            (f"hybrid_n{n}_lcf.csv",   f"hybrid_n{n}.grid.steady",  f"Hybrid (N={n})"),
        ]
        for lcf, prefix, label in runs:
            print(f"\n-- {label} --")
            run_pact(exp_dir, lcf, "experiment.config", "modelParams.config", prefix)
    else:
        print("=== Skipping PACT runs (--skip-sim) ===\n")

    print("\n=== Generating figures ===\n")

    steps = [
        ("1. Heatmap comparison (N=1)",          lambda: fig_heatmap_comparison(exp_dir, out_dir)),
        ("2. Peak temperature comparison (N=1)", lambda: fig_peak_temp_comparison(exp_dir, out_dir)),
        ("3. Block transient comparison (N=1)",  lambda: fig_block_transient_comparison(exp_dir, out_dir)),
        ("4. Thermal resistance breakdown",      lambda: fig_thermal_resistance(out_dir)),
        ("5. Power budget comparison",           lambda: fig_power_budget(exp_dir, out_dir)),
        ("6. Thermal scaling (N=1..8)",          lambda: fig_thermal_scaling(out_dir, n_max=8)),
        (f"7. Hybrid N={n} layer profile",       lambda: fig_hybrid_layer_profile(exp_dir, out_dir, n)),
        (f"8. Hybrid N={n} heatmaps",            lambda: fig_hybrid_heatmaps(exp_dir, out_dir, n)),
    ]
    for label, fn in steps:
        print(f"[{label}]")
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
