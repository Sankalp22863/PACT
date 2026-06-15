"""
NVDRAM Experiment — Thermal Visualization Suite
================================================
Generates publication-quality figures from PACT simulation outputs.

Usage:
    python generate_nvdram_diagrams.py [--exp-dir PATH] [--out-dir PATH]

Defaults:
    --exp-dir  ../NVDRAM_Experiment
    --out-dir  . (same folder as this script)

Figures generated:
    1. steady_state_heatmaps.png        — 40×40 temperature grid per layer (steady state)
    2. floorplan_power_overlay.png      — Chip floorplan with power-scenario annotations
    3. block_transient_temperatures.png — Block temperature evolution over simulation time
    4. cross_layer_peak_temperatures.png — Peak / mean / min temps across all layers
    5. transient_grid_snapshots.png     — Grid heatmap snapshots at 5 timesteps
    6. power_scenarios_comparison.png   — Grouped bar chart: idle / read / write power
    7. thermal_gradient_profiles.png    — Temperature profiles along X and Y chip axes
    8. temperature_distribution.png     — Box-violin distribution of temps per layer
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# ── aesthetic defaults ──────────────────────────────────────────────────────
CMAP_THERMAL  = "inferno"
CMAP_COOLWARM = "coolwarm"
DPI           = 150
FIGSIZE_WIDE  = (16, 5)
FIGSIZE_TALL  = (14, 10)
FONT_TITLE    = 13
FONT_AXIS     = 11

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.titlesize":     FONT_TITLE,
    "axes.labelsize":     FONT_AXIS,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.dpi":         DPI,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
})

KELVIN = 273.15  # offset for K → °C

LAYER_NAMES = {
    0: "CMOS (Layer 0)",
    1: "1T1C Mem Layer 1",
    2: "1T1C Mem Layer 2",
    3: "Heat Spreader (Layer 3)",
    4: "Heat Spreader (Layer 4)",
}

PTRACE_SCENARIOS = ["Idle", "Active Read", "Active Write"]


# ── helpers ─────────────────────────────────────────────────────────────────

def k2c(arr):
    """Kelvin → Celsius."""
    return np.asarray(arr, dtype=float) - KELVIN


def load_grid_steady(exp_dir, layer_id):
    """Load a .grid.steady.layerN file → 40×40 °C array."""
    path = os.path.join(exp_dir, f"nvdram.grid.steady.layer{layer_id}")
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path)            # 1600 flat values in Kelvin
    grid = k2c(data).reshape(40, 40)   # row-major
    return grid


def load_block_transient(exp_dir):
    """
    Parse nvdram.block.transient.csv.

    Returns dict:  {layer_id (int): DataFrame(columns=[block_name, step_index, temp_C])}
    """
    path = os.path.join(exp_dir, "nvdram.block.transient.csv")
    if not os.path.exists(path):
        return {}

    records = []
    current_step = None
    current_layer = None

    step_re  = re.compile(r"^step\s+(\d+)\s+layer number:(\d+)")
    layer_re = re.compile(r"^layer number:(\d+)")

    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            m = step_re.match(line)
            if m:
                current_step  = int(m.group(1))
                current_layer = int(m.group(2))
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
                    block_name = parts[1]
                    temp_c     = float(parts[2])
                    records.append((current_step, current_layer, block_name, temp_c))
                except (ValueError, IndexError):
                    pass

    if not records:
        return {}

    df = pd.DataFrame(records, columns=["step", "layer", "block", "temp_C"])
    return {lid: grp.reset_index(drop=True) for lid, grp in df.groupby("layer")}


def load_cir_transient(exp_dir, n_rows=None):
    """
    Load nvdram.cir.csv.

    Returns (times_s, grids) where grids[layer] is list of 40×40 °C arrays.
    Pass n_rows to limit memory use (None = all).
    """
    path = os.path.join(exp_dir, "nvdram.cir.csv")
    if not os.path.exists(path):
        return None, {}

    df = pd.read_csv(path, nrows=n_rows)
    times = df["TIME"].values

    # Discover how many layers are present
    col_sample = [c for c in df.columns if c.startswith("V(NODE")]
    layer_ids = sorted({int(re.match(r"V\(NODE(\d+)_", c).group(1)) for c in col_sample})

    grids = {}
    for lid in layer_ids:
        layer_cols = sorted(
            [c for c in col_sample if c.startswith(f"V(NODE{lid}_")],
            key=lambda c: (int(c.split("_")[1]), int(c.split("_")[2].rstrip(")")))
        )
        if len(layer_cols) != 1600:
            continue
        data = k2c(df[layer_cols].values)  # shape (T, 1600)
        grids[lid] = [data[t].reshape(40, 40) for t in range(len(times))]

    return times, grids


def load_floorplan(exp_dir, flp_file):
    """Load a floorplan CSV → DataFrame."""
    path = os.path.join(exp_dir, flp_file)
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_ptrace(exp_dir, ptrace_file):
    """Load a power trace CSV → DataFrame."""
    path = os.path.join(exp_dir, ptrace_file)
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


# ── figure 1: steady-state heatmaps ────────────────────────────────────────

def fig_steady_heatmaps(exp_dir, out_dir):
    layers_data = [(lid, load_grid_steady(exp_dir, lid)) for lid in range(5)]
    layers_data = [(lid, g) for lid, g in layers_data if g is not None]
    if not layers_data:
        print("  [skip] no grid.steady files found")
        return

    n = len(layers_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    vmin = min(g.min() for _, g in layers_data)
    vmax = max(g.max() for _, g in layers_data)

    for ax, (lid, grid) in zip(axes, layers_data):
        im = ax.imshow(grid, origin="lower", cmap=CMAP_THERMAL,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(LAYER_NAMES.get(lid, f"Layer {lid}"), fontsize=FONT_TITLE)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")

    fig.colorbar(im, ax=axes, label="Temperature (°C)", fraction=0.02, pad=0.04)
    fig.suptitle("NVDRAM Steady-State Thermal Distribution (40×40 Grid)",
                 fontsize=FONT_TITLE + 1, fontweight="bold", y=1.01)
    out = os.path.join(out_dir, "steady_state_heatmaps.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 2: floorplan + power overlay ────────────────────────────────────

def fig_floorplan_power(exp_dir, out_dir):
    flp_cmos = load_floorplan(exp_dir, "nvdram_cmos_flp.csv")
    ptrace_cmos = load_ptrace(exp_dir, "nvdram_cmos_ptrace.csv")
    ptrace_mem  = load_ptrace(exp_dir, "nvdram_mem_ptrace.csv")

    if flp_cmos is None or ptrace_cmos is None:
        print("  [skip] floorplan/ptrace files not found")
        return

    # Rename columns for clarity
    ptrace_cmos.columns = ["UnitName"] + PTRACE_SCENARIOS
    ptrace_mem.columns  = ["UnitName"] + PTRACE_SCENARIOS

    palette = plt.get_cmap("tab10")
    block_colors = {name: palette(i) for i, name in enumerate(flp_cmos["UnitName"])}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NVDRAM Floorplan — CMOS Layer Power by Scenario",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for ax_idx, scenario in enumerate(PTRACE_SCENARIOS):
        ax = axes[ax_idx]
        ax.set_aspect("equal")
        ax.set_title(f"Scenario: {scenario}", fontsize=FONT_TITLE)

        merged = flp_cmos.merge(ptrace_cmos[["UnitName", scenario]], on="UnitName", how="left")
        # Build power density per unit area
        merged["area"] = merged["Length (m)"] * merged["Width (m)"]
        merged["pdens"] = merged[scenario] / merged["area"]

        p_norm = Normalize(vmin=merged["pdens"].min(), vmax=merged["pdens"].max())
        sm     = ScalarMappable(cmap="YlOrRd", norm=p_norm)

        for _, row in merged.iterrows():
            x, y = row["X"], row["Y"]
            w, h = row["Length (m)"], row["Width (m)"]
            color = sm.to_rgba(row["pdens"])
            rect  = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="square,pad=0",
                linewidth=1, edgecolor="black",
                facecolor=color, alpha=0.85,
            )
            ax.add_patch(rect)
            cx, cy = x + w / 2, y + h / 2
            ax.text(cx, cy,
                    f"{row['UnitName']}\n{row[scenario]:.2f}W",
                    ha="center", va="center", fontsize=7, wrap=True)

        chip_size = 0.0085
        ax.set_xlim(-0.0005, chip_size + 0.0005)
        ax.set_ylim(-0.0005, chip_size + 0.0005)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*1e3:.1f}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*1e3:.1f}"))
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.colorbar(sm, ax=ax, label="Power Density (W/m²)", fraction=0.046, pad=0.04)

    out = os.path.join(out_dir, "floorplan_power_overlay.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 3: block transient temperatures ─────────────────────────────────

def fig_block_transient(exp_dir, out_dir, step_size_s=0.00333):
    layer_data = load_block_transient(exp_dir)
    if not layer_data:
        print("  [skip] block transient file not found")
        return

    n_layers = len(layer_data)
    ncols    = min(n_layers, 3)
    nrows    = (n_layers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("NVDRAM Block Temperature Transient (per Layer)",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    palette = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (lid, df) in enumerate(sorted(layer_data.items())):
        ax   = axes[idx // ncols][idx % ncols]
        time = df["step"].unique()
        time_s = (time - 1) * step_size_s * 100  # ptrace_step = 100 × step_size

        for b_idx, block in enumerate(df["block"].unique()):
            sub    = df[df["block"] == block].sort_values("step")
            temps  = sub["temp_C"].values
            n_pts  = min(len(time_s), len(temps))
            ax.plot(time_s[:n_pts], temps[:n_pts],
                    label=block, color=palette[b_idx], linewidth=1.8)

        ax.set_title(LAYER_NAMES.get(lid, f"Layer {lid}"), fontsize=FONT_TITLE)
        ax.set_xlabel("Simulation Time (s)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(n_layers, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    out = os.path.join(out_dir, "block_transient_temperatures.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 4: cross-layer peak / mean / min temperatures ───────────────────

def fig_cross_layer_temps(exp_dir, out_dir):
    stats = []
    for lid in range(5):
        grid = load_grid_steady(exp_dir, lid)
        if grid is not None:
            stats.append({
                "layer":   LAYER_NAMES.get(lid, f"Layer {lid}"),
                "peak":    grid.max(),
                "mean":    grid.mean(),
                "min":     grid.min(),
                "std":     grid.std(),
            })

    if not stats:
        print("  [skip] no steady-state data for cross-layer comparison")
        return

    df  = pd.DataFrame(stats)
    x   = np.arange(len(df))
    w   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_peak = ax.bar(x - w,   df["peak"], w, label="Peak",  color="#d62728", alpha=0.85)
    bars_mean = ax.bar(x,       df["mean"], w, label="Mean",  color="#2ca02c", alpha=0.85)
    bars_min  = ax.bar(x + w,   df["min"],  w, label="Min",   color="#1f77b4", alpha=0.85)

    # error bars for std on mean
    ax.errorbar(x, df["mean"], yerr=df["std"], fmt="none", color="black",
                capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["layer"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Steady-State Temperature Statistics Across Chip Layers",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # annotate peak values
    for bar in bars_peak:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#d62728")

    out = os.path.join(out_dir, "cross_layer_peak_temperatures.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 5: transient grid snapshots ─────────────────────────────────────

def fig_transient_snapshots(exp_dir, out_dir, n_snapshots=5):
    times, grids = load_cir_transient(exp_dir)
    if not grids:
        print("  [skip] cir transient file not found or unreadable")
        return

    layer_ids = sorted(grids.keys())
    n_layers  = len(layer_ids)
    n_steps   = len(times)
    snap_idx  = [int(i * (n_steps - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]

    # global colour range
    all_temps = np.concatenate([grids[lid][t].ravel()
                                for lid in layer_ids for t in snap_idx])
    vmin, vmax = all_temps.min(), all_temps.max()

    fig, axes = plt.subplots(n_layers, n_snapshots,
                             figsize=(3 * n_snapshots, 3 * n_layers))
    fig.suptitle("NVDRAM Transient Grid Temperature Snapshots",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for row, lid in enumerate(layer_ids):
        for col, t_idx in enumerate(snap_idx):
            ax = axes[row][col] if n_layers > 1 else axes[col]
            im = ax.imshow(grids[lid][t_idx], origin="lower",
                           cmap=CMAP_THERMAL, vmin=vmin, vmax=vmax, aspect="equal")
            t_ms = times[t_idx] * 1e3
            if row == 0:
                ax.set_title(f"t = {t_ms:.1f} ms", fontsize=9)
            if col == 0:
                ax.set_ylabel(LAYER_NAMES.get(lid, f"L{lid}"), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.colorbar(im, ax=axes.ravel().tolist() if hasattr(axes, "ravel") else axes,
                 label="Temperature (°C)", fraction=0.015, pad=0.02)
    out = os.path.join(out_dir, "transient_grid_snapshots.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 6: power scenarios comparison ───────────────────────────────────

def fig_power_scenarios(exp_dir, out_dir):
    ptrace_cmos = load_ptrace(exp_dir, "nvdram_cmos_ptrace.csv")
    ptrace_mem  = load_ptrace(exp_dir, "nvdram_mem_ptrace.csv")
    if ptrace_cmos is None:
        print("  [skip] ptrace files not found")
        return

    ptrace_cmos.columns = ["UnitName"] + PTRACE_SCENARIOS
    ptrace_mem.columns  = ["UnitName"] + PTRACE_SCENARIOS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Power Consumption by Scenario and Functional Block",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    for ax, (ptrace, title) in zip(axes, [
        (ptrace_cmos, "CMOS Layer"),
        (ptrace_mem,  "1T1C Memory Layer"),
    ]):
        x      = np.arange(len(ptrace))
        n_scen = len(PTRACE_SCENARIOS)
        width  = 0.22
        colors = ["#4878cf", "#6acc65", "#d65f5f"]

        for s_idx, (scen, color) in enumerate(zip(PTRACE_SCENARIOS, colors)):
            offset = (s_idx - n_scen / 2 + 0.5) * width
            bars   = ax.bar(x + offset, ptrace[scen], width,
                            label=scen, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(ptrace["UnitName"], rotation=15, ha="right")
        ax.set_ylabel("Power (W)")
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(out_dir, "power_scenarios_comparison.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 7: thermal gradient profiles ────────────────────────────────────

def fig_thermal_gradient_profiles(exp_dir, out_dir):
    layers_data = [(lid, load_grid_steady(exp_dir, lid)) for lid in range(3)]
    layers_data = [(lid, g) for lid, g in layers_data if g is not None]
    if not layers_data:
        print("  [skip] no steady-state data for gradient profiles")
        return

    chip_size_mm = 8.5
    positions    = np.linspace(0, chip_size_mm, 40)
    mid          = 20  # centre row/col index

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("NVDRAM Steady-State Thermal Gradient Profiles (Centre Slice)",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    palette = plt.cm.Set2(np.linspace(0, 0.9, len(layers_data)))

    for ax, direction in zip(axes, ["X (column)", "Y (row)"]):
        for (lid, grid), color in zip(layers_data, palette):
            if direction.startswith("X"):
                profile = grid[mid, :]    # row = mid, all columns
            else:
                profile = grid[:, mid]    # all rows, col = mid
            ax.plot(positions, profile, label=LAYER_NAMES.get(lid, f"L{lid}"),
                    color=color, linewidth=2)

        ax.set_xlabel(f"Position along {direction} (mm)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"Profile along {direction} axis", fontsize=FONT_TITLE)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    out = os.path.join(out_dir, "thermal_gradient_profiles.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── figure 8: temperature distribution (violin + box) ──────────────────────

def fig_temperature_distribution(exp_dir, out_dir):
    records = []
    for lid in range(5):
        grid = load_grid_steady(exp_dir, lid)
        if grid is not None:
            for val in grid.ravel():
                records.append({"Layer": LAYER_NAMES.get(lid, f"Layer {lid}"), "Temperature (°C)": val})

    if not records:
        print("  [skip] no steady-state data for distribution plot")
        return

    df  = pd.DataFrame(records)
    layers_order = df["Layer"].unique().tolist()
    data_by_layer = [df.loc[df["Layer"] == lname, "Temperature (°C)"].values
                     for lname in layers_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(layers_order)))

    bp = ax.boxplot(data_by_layer, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # jitter overlay
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data_by_layer):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color="black", alpha=0.08, s=4, zorder=3)

    ax.set_xticks(range(1, len(layers_order) + 1))
    ax.set_xticklabels(layers_order, rotation=20, ha="right")
    ax.set_title("Steady-State Temperature Distribution per Chip Layer",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(out_dir, "temperature_distribution.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate NVDRAM thermal diagrams.")
    parser.add_argument("--exp-dir", default=os.path.join(os.path.dirname(__file__), "..", "NVDRAM_Experiment"),
                        help="Path to the NVDRAM_Experiment folder")
    parser.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for PNG figures")
    args = parser.parse_args()

    exp_dir = os.path.realpath(args.exp_dir)
    out_dir = os.path.realpath(args.out_dir)

    if not os.path.isdir(exp_dir):
        sys.exit(f"ERROR: experiment directory not found: {exp_dir}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Experiment dir : {exp_dir}")
    print(f"Output dir     : {out_dir}\n")

    steps = [
        ("1. Steady-state heatmaps",          fig_steady_heatmaps),
        ("2. Floorplan + power overlay",       fig_floorplan_power),
        ("3. Block transient temperatures",    fig_block_transient),
        ("4. Cross-layer temperature stats",   fig_cross_layer_temps),
        ("5. Transient grid snapshots",        fig_transient_snapshots),
        ("6. Power scenarios comparison",      fig_power_scenarios),
        ("7. Thermal gradient profiles",       fig_thermal_gradient_profiles),
        ("8. Temperature distribution",        fig_temperature_distribution),
    ]

    for label, fn in steps:
        print(f"[{label}]")
        try:
            fn(exp_dir, out_dir)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
