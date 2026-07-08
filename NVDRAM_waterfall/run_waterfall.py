"""
phi-HBM STCO Waterfall driver  (NVDRAM_waterfall)
===================================================
Runs the phi-HBM thermal waterfall: starting from the mold-filled BASELINE
(EXP_NVDRAM_B regime, no thermal silicon) it applies the imec STCO interventions
CUMULATIVELY, one per stage, running a PACT steady-state solve at each stage and
recording the peak compute-die (GPU) temperature.

Requested sequence (each stage adds one optimization on top of the previous):
    0  Baseline (3D stacking)      mold filler, base die, thick top die, 414 W
    1  + HBM base-die removal      --no-base-die
    2  + HBM stack merging         --merge-stacks
    3  + Top-die thinning          --thin-top-die
    4  + 0.7x GPU frequency        --gpu-power 289.8   (0.7 * 414, linear P~f)
    5  + Thermal-silicon opt.      --optimized         (mold -> thermal silicon)
(2-sided cooling intentionally left out for now.)

Peak GPU temperature = max over the GPU FEOL layer UNDER the 2x2 memory stacks
(the framework's mask_surround convention: the mold gap between stacks cooks to a
few-hundred C as an artifact of the thin uniform-power GPU layer, so it is
excluded; heat actually escapes under the stacks).

Each stage runs in its own subdirectory (stage0/ ... stage5/) so all inputs and
grid outputs are preserved. Results -> waterfall_results.csv + waterfall_log.txt.

Usage:  python3 run_waterfall.py
"""

import argparse
import os
import shlex
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACT = os.path.abspath(os.path.join(HERE, "..", "src", "PACT.py"))
GEN  = os.path.join(HERE, "waterfall_generate.py")

GPU_SIDE = 0.024
MEM_SIDE = 0.010
GPU_FEOL_LAYER = 1          # layer index of the GPU compute-die heat source
GPU_POWER_FULL = 414.0
FREQ_SCALE = 0.7

# Cumulative flag sets. Each stage = previous flags + one new intervention.
STAGES = [
    ("0_baseline",      "Baseline (3D stacking)",        []),
    ("1_base_removal",  "+ HBM base-die removal",        ["--no-base-die"]),
    ("2_merge",         "+ HBM stack merging",           ["--no-base-die", "--merge-stacks"]),
    ("3_thin_top",      "+ Top-die thinning",            ["--no-base-die", "--merge-stacks", "--thin-top-die"]),
    ("4_freq_0p7",      "+ 0.7x GPU frequency",          ["--no-base-die", "--merge-stacks", "--thin-top-die",
                                                          f"--gpu-power {GPU_POWER_FULL*FREQ_SCALE:g}"]),
    ("5_thermal_si",    "+ Thermal-silicon optimization", ["--no-base-die", "--merge-stacks", "--thin-top-die",
                                                          f"--gpu-power {GPU_POWER_FULL*FREQ_SCALE:g}", "--optimized"]),
]

GRID_PREFIX = "hybrid.grid.steady"


def under_stack_peak(grid_path, gpu_side=GPU_SIDE, mem_side=MEM_SIDE):
    """Max/min GPU temperature (C) UNDER the 2x2 memory stacks, plus raw grid
    min/max. Replicates pact_plots.under_stack_range (mask_surround convention)."""
    data = np.loadtxt(grid_path) - 273.15          # K -> C
    side = int(round(np.sqrt(data.size)))
    g = data.reshape(side, side)
    gap = gpu_side - 2 * mem_side
    margin, central = gap / 4.0, gap / 2.0
    a = [margin, margin + mem_side, margin + mem_side + central, margin + 2 * mem_side + central]
    f = [int(round(x / gpu_side * side)) for x in a]
    (lo1, hi1), (lo2, hi2) = (f[0], f[1]), (f[2], f[3])
    m = np.zeros(g.shape, dtype=bool)
    for x0, x1 in ((lo1, hi1), (lo2, hi2)):
        for y0, y1 in ((lo1, hi1), (lo2, hi2)):
            m[y0:y1, x0:x1] = True
    vals = g[m]
    return float(vals.max()), float(vals.min()), float(g.min()), float(g.max())


def run(cmd, cwd):
    subprocess.run(shlex.split(cmd) if isinstance(cmd, str) else cmd,
                   cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description="Run the phi-HBM STCO waterfall for one stack composition.")
    ap.add_argument("--nv-tiers", type=int, default=4, help="NVDRAM tiers (bottom); 0 = all-DRAM")
    ap.add_argument("--dram-tiers", type=int, default=8, help="DRAM tiers (top)")
    ap.add_argument("--label", default="φ-HBM: 4 NVDRAM + 8 DRAM",
                    help="human-readable stack label (used in the figure subtitle)")
    args = ap.parse_args()
    comp = [f"--nv-tiers {args.nv_tiers}", f"--dram-tiers {args.dram_tiers}"]

    with open(os.path.join(HERE, "waterfall_meta.txt"), "w") as fh:
        fh.write(args.label + "\n")

    rows = []
    log_lines = []
    prev_peak = None
    for sid, label, flags in STAGES:
        sdir = os.path.join(HERE, sid)
        os.makedirs(sdir, exist_ok=True)
        for cfg in ("experiment.config", "modelParams.config"):
            src = os.path.join(HERE, cfg)
            with open(src) as a, open(os.path.join(sdir, cfg), "w") as b:
                b.write(a.read())

        # 1. generate inputs for this stage (composition prepended to the stage flags)
        run(f"python3 {shlex.quote(GEN)} " + " ".join(comp + flags), cwd=sdir)
        # 2. PACT steady-state solve
        run(f"python3 {shlex.quote(PACT)} hybrid_lcf.csv experiment.config "
            f"modelParams.config --gridSteadyFile {GRID_PREFIX}", cwd=sdir)
        # 3. extract peak GPU temperature (under-stack)
        grid = os.path.join(sdir, f"{GRID_PREFIX}.layer{GPU_FEOL_LAYER}")
        peak, umin, gmin, gmax = under_stack_peak(grid)
        delta = "" if prev_peak is None else f"{peak - prev_peak:+.2f}"
        gpu_power = GPU_POWER_FULL * FREQ_SCALE if "--gpu-power" in " ".join(flags) else GPU_POWER_FULL
        rows.append((sid, label, f"{peak:.2f}", delta, f"{gpu_power:g}",
                     f"{umin:.2f}", f"{gmax:.2f}"))
        step = f"drop {prev_peak - peak:+.2f} C" if prev_peak is not None else "start"
        line = (f"[{sid:14s}] {label:34s} peak GPU (under-stack) = {peak:7.2f} C   "
                f"({step}; raw grid {gmin:.1f}..{gmax:.1f} C)")
        print(line)
        log_lines.append(line)
        prev_peak = peak

    # write results CSV
    csv_path = os.path.join(HERE, "waterfall_results.csv")
    with open(csv_path, "w") as fh:
        fh.write("stage,label,peak_gpu_understack_C,delta_C,gpu_power_W,understack_min_C,raw_grid_max_C\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
    with open(os.path.join(HERE, "waterfall_log.txt"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")

    print(f"\nWaterfall complete. Peak GPU temperature by stage:")
    print(f"  {rows[0][2]} C (baseline)  ->  {rows[-1][2]} C (all optimizations)")
    print(f"  total drop: {float(rows[0][2]) - float(rows[-1][2]):+.2f} C")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
