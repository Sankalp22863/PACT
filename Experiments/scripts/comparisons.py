"""
Cross-experiment comparison figures for the PACT 3D-stack sweeps.
=================================================================
Parses each experiment (via its meta.json: grid_prefix + lcf + generated
ptrace files) and renders the figures that only make sense *across* the
NVDRAM/DRAM stack-composition sweep:

  * ram_tier_temps      - peak temperature of every memory die up the stack,
                          one line per stack composition (paper's "peak temp
                          of each HBM die" figure).
  * peak_vs_nv_fraction - GPU vs memory peak temperature, and the chip power
                          that drives it, as a function of how many tiers are
                          NVDRAM (1x2 small-multiple, one shared y per panel).

Memory Si dies are the LCF rows whose floorplan is nv_tier_flp.csv (NVDRAM) or
dram_tier_flp.csv (DRAM); the HBM base die is hbm_base_si_flp.csv and the GPU
compute die is gpu_feol_flp.csv. Powers are summed from the referenced ptrace
files, so the numbers are correct for any experiment's power model.

`make_comparisons()` discovers the experiments and renders everything; the
individual functions are importable on their own too.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import pact_plots  # noqa: E402  (shared load_grid / read_lcf helpers)

MEM_FLP = {"nv_tier_flp.csv": "NVDRAM", "dram_tier_flp.csv": "DRAM"}
BASE_FLP = "hbm_base_si_flp.csv"
GPU_FLP = "gpu_feol_flp.csv"

# Fixed categorical order, CVD-validated (Okabe-Ito subset). Never cycled.
SERIES_COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
KIND_MARKER = {"NVDRAM": "o", "DRAM": "s", "Base": "D"}
GPU_COLOR, MEM_COLOR, TOTAL_COLOR = "#D55E00", "#0072B2", "#333333"


def _ptrace_power(exp_dir, ptrace_file):
    """Sum the steady 'Power' column of a ptrace file (0 if missing)."""
    path = os.path.join(exp_dir, ptrace_file)
    if not os.path.isfile(path):
        return 0.0
    total = 0.0
    with open(path) as fh:
        next(fh, None)                       # header
        for line in fh:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass
    return total


def tier_profile(exp_dir):
    """Per-experiment profile, or None if meta/grids are missing.

    {'dir','name','base','tiers':[(kind,peak_C)..bottom->top],'n_nv','n_dram',
     'gpu' (peak_C), 'gpu_power','nv_power','dram_power','mem_power','total_power'}
    """
    meta_path = os.path.join(exp_dir, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"  (skip {exp_dir}: no meta.json)")
        return None
    meta = json.load(open(meta_path))
    prefix = os.path.join(exp_dir, meta["grid_prefix"])
    lcf = os.path.join(exp_dir, meta["lcf"])
    if not os.path.isfile(lcf) or not glob.glob(f"{prefix}.layer*"):
        print(f"  (skip {exp_dir}: no LCF or grid files -- run the experiment first)")
        return None

    base, gpu, tiers = None, None, []
    gpu_power = nv_power = dram_power = 0.0
    for layer_id, flp, _ in pact_plots.read_lcf(lcf):
        # power (from the ptrace referenced for this layer)
        ptrace = _lcf_ptrace(lcf, layer_id)
        if flp == GPU_FLP:
            g = pact_plots.load_grid(prefix, layer_id)
            if g is not None:
                gpu = float(g.max())
            gpu_power += _ptrace_power(exp_dir, ptrace) if ptrace else 0.0
        elif flp == BASE_FLP:
            g = pact_plots.load_grid(prefix, layer_id)
            if g is not None:
                base = float(g.max())
        elif flp in MEM_FLP:
            g = pact_plots.load_grid(prefix, layer_id)
            if g is None:
                print(f"  (skip {exp_dir}: missing grid for layer {layer_id})")
                return None
            tiers.append((MEM_FLP[flp], float(g.max())))
            p = _ptrace_power(exp_dir, ptrace) if ptrace else 0.0
            if MEM_FLP[flp] == "NVDRAM":
                nv_power += p
            else:
                dram_power += p

    if not tiers:
        print(f"  (skip {exp_dir}: no memory tiers found)")
        return None
    n_nv = sum(1 for k, _ in tiers if k == "NVDRAM")
    n_dram = sum(1 for k, _ in tiers if k == "DRAM")
    mem_power = nv_power + dram_power
    return {"dir": exp_dir, "name": meta.get("name", os.path.basename(exp_dir)),
            "base": base, "gpu": gpu, "tiers": tiers, "n_nv": n_nv, "n_dram": n_dram,
            "gpu_power": gpu_power, "nv_power": nv_power, "dram_power": dram_power,
            "mem_power": mem_power, "total_power": gpu_power + mem_power}


# small cache so we read each LCF once for the ptrace column
_LCF_PTRACE = {}


def _lcf_ptrace(lcf_path, layer_id):
    """Return the PtraceFile column for a given layer id in an LCF."""
    table = _LCF_PTRACE.get(lcf_path)
    if table is None:
        table = {}
        with open(lcf_path) as fh:
            next(fh, None)
            for line in fh:
                parts = [c.strip() for c in line.split(",")]
                if len(parts) >= 4:
                    table[int(parts[0])] = parts[3]
        _LCF_PTRACE[lcf_path] = table
    return table.get(layer_id)


def series_label(p):
    tag = " (pure-DRAM baseline)" if p["n_nv"] == 0 else ""
    return f"{p['n_nv']} NV + {p['n_dram']} DRAM{tag}"


# ── chart 1: per-memory-tier peak temperature ────────────────────────────────
def ram_tier_temps(profiles, out):
    if not profiles:
        raise SystemExit("ram_tier_temps: no experiments to plot.")
    if len(profiles) > len(SERIES_COLORS):
        raise SystemExit(f"Only {len(SERIES_COLORS)} categorical colors defined; "
                         f"got {len(profiles)} series.")
    n_tiers = max(len(p["tiers"]) for p in profiles)
    x_tier = np.arange(1, n_tiers + 1)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for p, color in zip(profiles, SERIES_COLORS):
        temps = [t for _, t in p["tiers"]]
        kinds = [k for k, _ in p["tiers"]]
        xs, ys = list(x_tier[:len(temps)]), list(temps)
        if p["base"] is not None:
            xs, ys, kinds = [0] + xs, [p["base"]] + ys, ["Base"] + kinds
        ax.plot(xs, ys, "-", color=color, lw=2.0, zorder=2, label=series_label(p))
        for x, y, k in zip(xs, ys, kinds):
            ax.plot(x, y, KIND_MARKER[k], color=color, ms=8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.annotate(f"{temps[0]:.1f}°C", (x_tier[0], temps[0]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8.5, fontweight="bold", color=color)
    xticks = ([0] if any(p["base"] is not None for p in profiles) else []) + list(x_tier)
    xlabels = (["Base"] if 0 in xticks else []) + [str(i) for i in x_tier]
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
    ax.set_xlabel("Memory die position   (Base = nearest GPU  →  top tier = nearest cooling lid)")
    ax.set_ylabel("Peak die temperature (°C)")
    ax.set_title("Peak temperature of each memory die up the 3D stack\n"
                 "NVDRAM (bottom) + DRAM (top) — varying stack composition", fontsize=12)
    ax.grid(True, ls=":", alpha=0.5)
    series_leg = ax.legend(title="Stack composition", loc="upper right", fontsize=9)
    ax.add_artist(series_leg)
    marker_handles = [
        Line2D([], [], marker="D", color="0.35", ls="none", ms=8, label="HBM base die"),
        Line2D([], [], marker="o", color="0.35", ls="none", ms=8, label="NVDRAM tier"),
        Line2D([], [], marker="s", color="0.35", ls="none", ms=8, label="DRAM tier"),
    ]
    ax.legend(handles=marker_handles, title="Die type", loc="lower left", fontsize=9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    Written: {out}")


# ── chart 2: GPU vs memory peak (and the power that drives it) vs NVDRAM count ─
def peak_vs_nv_fraction(profiles, out):
    if len(profiles) < 2:
        print("    (peak_vs_nv_fraction: need >=2 experiments; skipping)")
        return
    ps = sorted(profiles, key=lambda p: p["n_nv"])
    x = [p["n_nv"] for p in ps]
    total = ps[0]["n_nv"] + ps[0]["n_dram"]
    gpu = [p["gpu"] for p in ps]
    mem_hot = [max(t for _, t in p["tiers"]) for p in ps]
    mem_cold = [min(t for _, t in p["tiers"]) for p in ps]

    fig, (axT, axP) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # -- panel A: peak temperature --
    axT.fill_between(x, mem_cold, mem_hot, color=MEM_COLOR, alpha=0.12, zorder=0,
                     label="memory range (top→bottom die)")
    axT.plot(x, gpu, "o-", color=GPU_COLOR, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=0.8, label="GPU compute die (peak)")
    axT.plot(x, mem_hot, "s-", color=MEM_COLOR, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=0.8, label="hottest memory die (bottom)")
    for xi, g, m in zip(x, gpu, mem_hot):
        if xi in (x[0], x[-1]):
            axT.annotate(f"{g:.1f}°C", (xi, g), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color=GPU_COLOR)
            axT.annotate(f"{m:.1f}°C", (xi, m), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color=MEM_COLOR)
    axT.set_xticks(x); axT.set_xticklabels([f"{xi}\n({xi/total:.0%})" for xi in x])
    axT.set_xlabel("NVDRAM tiers in the memory stack  (fraction of 12)")
    axT.set_ylabel("Peak temperature (°C)")
    axT.set_title("GPU vs memory peak temperature")
    axT.grid(True, ls=":", alpha=0.5)
    axT.legend(fontsize=9, loc="upper right")

    # -- panel B: chip power (the driver) --
    total_p = [p["total_power"] for p in ps]
    mem_p = [p["mem_power"] for p in ps]
    gpu_p = ps[0]["gpu_power"]
    axP.plot(x, total_p, "o-", color=TOTAL_COLOR, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=0.8, label="total chip power")
    axP.plot(x, mem_p, "s-", color=MEM_COLOR, lw=2.2, ms=8,
             markeredgecolor="white", markeredgewidth=0.8, label="memory standby power")
    axP.axhline(gpu_p, color="0.5", ls="--", lw=1.4, zorder=1)
    axP.annotate(f"GPU heat source (fixed {gpu_p:.0f} W)", (x[0], gpu_p),
                 textcoords="offset points", xytext=(4, 6), fontsize=8.5, color="0.4")
    for xi, tp, mp in zip(x, total_p, mem_p):
        if xi in (x[0], x[-1]):
            axP.annotate(f"{tp:.0f} W", (xi, tp), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold", color=TOTAL_COLOR)
            axP.annotate(f"{mp:.0f} W", (xi, mp), textcoords="offset points",
                         xytext=(0, -14), ha="center", fontsize=8.5, fontweight="bold", color=MEM_COLOR)
    axP.set_xticks(x); axP.set_xticklabels([f"{xi}\n({xi/total:.0%})" for xi in x])
    axP.set_xlabel("NVDRAM tiers in the memory stack  (fraction of 12)")
    axP.set_ylabel("Power (W)")
    axP.set_title("Chip power vs NVDRAM fraction")
    axP.grid(True, ls=":", alpha=0.5)
    axP.legend(fontsize=9, loc="center right")

    fig.suptitle("More NVDRAM → less memory standby power → cooler GPU and memory\n"
                 "(top-only cooling: all heat exits up through the lid)", fontsize=12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"    Written: {out}")


# ── discovery + dispatcher ────────────────────────────────────────────────────
def discover_profiles(exp_root=EXP_ROOT, baseline="EXP_DRAM_B"):
    """Discover EXP_NVDRAM_<n> experiments (+ optional pure-DRAM baseline).
    Variant dirs like EXP_NVDRAM_4_STCO are excluded — the NV-fraction sweep
    compares like-for-like baseline-conditions experiments only."""
    dirs = sorted(d for d in glob.glob(os.path.join(exp_root, "EXP_NVDRAM_[0-9]*"))
                  if re.fullmatch(r"EXP_NVDRAM_\d+", os.path.basename(d)))
    profiles = [p for p in (tier_profile(d) for d in dirs) if p]
    profiles.sort(key=lambda p: p["n_nv"])
    if baseline:
        b = tier_profile(os.path.join(exp_root, baseline))
        if b:
            profiles.insert(0, b)
    return profiles


def make_comparisons(exp_root=EXP_ROOT, out_dir=None, baseline="EXP_DRAM_B"):
    """Render all cross-experiment comparison figures into out_dir."""
    out_dir = out_dir or os.path.join(exp_root, "comparisons")
    profiles = discover_profiles(exp_root, baseline)
    if len(profiles) < 2:
        print(f"  (comparisons: found {len(profiles)} experiment(s); need >=2 -- skipping)")
        return []
    print(f"  Comparison figures across {len(profiles)} experiment(s): "
          + ", ".join(series_label(p) for p in profiles))
    ram_tier_temps(profiles, os.path.join(out_dir, "ram_tier_temperatures.png"))
    peak_vs_nv_fraction(profiles, os.path.join(out_dir, "peak_vs_nv_fraction.png"))
    return profiles
