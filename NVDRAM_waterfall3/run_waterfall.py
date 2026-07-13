"""
All-DRAM STCO waterfall driver — RECTANGULAR 30x22 die  (NVDRAM_baseline2)
=========================================================================
Runs the all-DRAM (12 DRAM) STCO waterfall on the rectangular 30 x 22 mm GPU die,
recording at EACH stage BOTH:
  * the peak compute-die (GPU) temperature, and
  * the peak of the BOTTOM-MOST DRAM tier (nearest the GPU),
each as the max under the 2x2 memory stacks (mask_surround convention).

Frequency stage is 0.5x (GPU power 0.5 * 414 = 207 W, linear P~f). Note: the imec
paper's 0.5x-frequency point is 300 W (only compute dynamic power scales); use
--gpu-power 300 at that stage for exact paper-matching.

Each stage runs in stageN/. Results -> waterfall_results.csv (gpu + bottom-DRAM),
waterfall_log.txt, waterfall_meta.txt.

Usage:  python3 run_waterfall.py [--nv-tiers 0] [--dram-tiers 12] [--label ...]
"""

import argparse
import os
import shlex
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACT = os.path.abspath(os.path.join(HERE, "..", "src", "PACT.py"))
GEN  = os.path.join(HERE, "waterfall_generate.py")

# rectangular geometry (must match waterfall_generate.py defaults + modelParams grid)
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT, CENTRAL_VOID = 0.001, 0.005      # paper-layout: edge inserts + merge void (matches VOID_UM)
GPU_POWER_FULL = 355.4925              # power density 5.38625e-7 W/um^2 x (30000 x 22000 um) = 355.49 W
FREQ_POWER = GPU_POWER_FULL * 346.0 / 414.0   # 0.7x frequency: paper Fig.8 gives 346 W at 0.7x (414 W full);
                                              # scaled to this GPU's full power -> 297 W (non-linear: L2/interconnect fixed)
SUBSTRATE_UM = 1500                    # GPU-Si substrate (calibration knob, same as NVDRAM_baseline2)
VOID_UM = 5000                         # effective mold-void width merge converts (same calibration)
GRID_PREFIX = "hybrid.grid.steady"

STAGES = [
    ("0_baseline",     "Baseline (3D stacking)",         []),
    ("1_base_removal", "+ HBM base-die removal",         ["--no-base-die"]),
    ("2_merge",        "+ HBM stack merging",            ["--no-base-die", "--merge-stacks"]),
    ("3_thin_top",     "+ Top-die thinning",             ["--no-base-die", "--merge-stacks", "--thin-top-die"]),
    ("4_freq_0p7",     "+ 0.7x GPU frequency",           ["--no-base-die", "--merge-stacks", "--thin-top-die",
                                                          f"--gpu-power {FREQ_POWER:g}"]),
    ("5_thermal_si",   "+ Thermal-silicon optimization", ["--no-base-die", "--merge-stacks", "--thin-top-die",
                                                          f"--gpu-power {FREQ_POWER:g}", "--optimized"]),
]


def stack_mask():
    """Boolean (ROWS, COLS) mask of the memory-stack footprints: the two edge
    columns (full die height) of the paper-layout floorplan."""
    gl = GPU_X / COLS
    cw = (GPU_X - 2 * EDGE_INSERT - CENTRAL_VOID) / 2.0
    x_left = (EDGE_INSERT, EDGE_INSERT + cw)
    x_right = (EDGE_INSERT + cw + CENTRAL_VOID, EDGE_INSERT + cw + CENTRAL_VOID + cw)
    m = np.zeros((ROWS, COLS), dtype=bool)
    for x0, x1 in (x_left, x_right):
        m[:, round(x0 / gl):round(x1 / gl)] = True
    return m


MASK = stack_mask()


def global_peak(grid_path):
    """Max temperature over the whole GPU layer (the central mold void is the
    physical hotspot in the paper layout, so no masking)."""
    g = (np.loadtxt(grid_path) - 273.15).reshape(ROWS, COLS)
    return float(g.max())


def under_stack_peak(grid_path):
    """Max temperature under the memory stacks (for the DRAM tier, which only
    exists under the stack columns)."""
    g = (np.loadtxt(grid_path) - 273.15).reshape(ROWS, COLS)
    return float(g[MASK].max())


def first_layer_with(lcf_path, flp_name):
    """Layer index of the first (lowest) layer whose floorplan == flp_name."""
    with open(lcf_path) as fh:
        next(fh, None)
        for line in fh:
            parts = [c.strip() for c in line.split(",")]
            if len(parts) >= 2 and parts[1] == flp_name:
                return int(parts[0])
    raise SystemExit(f"No {flp_name} layer in {lcf_path}")


def run(cmd, cwd):
    subprocess.run(shlex.split(cmd), cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nv-tiers", type=int, default=4)
    ap.add_argument("--dram-tiers", type=int, default=8)
    ap.add_argument("--label", default="φ-HBM: 4 NVDRAM + 8 DRAM (30x22)")
    args = ap.parse_args()
    comp = [f"--nv-tiers {args.nv_tiers}", f"--dram-tiers {args.dram_tiers}",
            f"--gpu-si-um {SUBSTRATE_UM}", f"--void-um {VOID_UM}", "--active"]
    open(os.path.join(HERE, "waterfall_meta.txt"), "w").write(args.label + "\n")

    rows, log = [], []
    prev_gpu = None
    bdram_layer_final = None
    gpu_layer_final = None
    nv_layer_final = None
    for sid, label, flags in STAGES:
        sdir = os.path.join(HERE, sid)
        os.makedirs(sdir, exist_ok=True)
        for cfg in ("experiment.config", "modelParams.config"):
            open(os.path.join(sdir, cfg), "w").write(open(os.path.join(HERE, cfg)).read())

        run(f"python3 {shlex.quote(GEN)} " + " ".join(comp + flags), cwd=sdir)
        run(f"python3 {shlex.quote(PACT)} hybrid_lcf.csv experiment.config "
            f"modelParams.config --gridSteadyFile {GRID_PREFIX}", cwd=sdir)

        lcf = os.path.join(sdir, "hybrid_lcf.csv")
        gpu_layer = first_layer_with(lcf, "gpu_feol_flp.csv")
        bd_layer = first_layer_with(lcf, "dram_tier_flp.csv")
        nv_layer = first_layer_with(lcf, "nv_tier_flp.csv")   # lowest NVDRAM tier (nearest GPU)
        bdram_layer_final = bd_layer
        gpu_layer_final = gpu_layer
        nv_layer_final = nv_layer
        gpu = global_peak(os.path.join(sdir, f"{GRID_PREFIX}.layer{gpu_layer}"))
        bdram = under_stack_peak(os.path.join(sdir, f"{GRID_PREFIX}.layer{bd_layer}"))
        gpu_pwr = FREQ_POWER if "--gpu-power" in " ".join(flags) else GPU_POWER_FULL
        d = "" if prev_gpu is None else f"{gpu - prev_gpu:+.2f}"
        rows.append((sid, label, f"{gpu:.2f}", d, f"{bdram:.2f}", f"{gpu_pwr:g}", str(bd_layer)))
        line = (f"[{sid:14s}] {label:34s} GPU {gpu:7.2f} C | bottom-DRAM(L{bd_layer}) {bdram:7.2f} C"
                f"{'' if prev_gpu is None else f'   (GPU {gpu-prev_gpu:+.2f})'}")
        print(line); log.append(line)
        prev_gpu = gpu

    with open(os.path.join(HERE, "waterfall_results.csv"), "w") as fh:
        fh.write("stage,label,peak_gpu_C,delta_gpu_C,bottom_dram_C,gpu_power_W,bottom_dram_layer\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
    open(os.path.join(HERE, "waterfall_log.txt"), "w").write("\n".join(log) + "\n")
    # record the post-thermal-Si GPU-FEOL and bottom-DRAM layers for the heatmap step
    open(os.path.join(HERE, "bottom_dram_layer.txt"), "w").write(str(bdram_layer_final) + "\n")
    open(os.path.join(HERE, "gpu_feol_layer.txt"), "w").write(str(gpu_layer_final) + "\n")
    open(os.path.join(HERE, "nv_tier_layer.txt"), "w").write(str(nv_layer_final) + "\n")

    print(f"\nDone. GPU {rows[0][2]} -> {rows[-1][2]} C ; "
          f"bottom-DRAM {rows[0][4]} -> {rows[-1][4]} C. Results: waterfall_results.csv")


if __name__ == "__main__":
    main()
