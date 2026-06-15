"""
NVDRAM-on-GPU 3D Stack Generator
================================
Generates the PACT input files for an imec-style 3D stack where an N-tier
NVDRAM (ferroelectric 1T1C) memory is integrated on top of a GPU compute die.

Stack, bottom (package side, layer 0) -> top (heat-sink side):

    GPU substrate/TSV  ->  GPU FEOL/TSV  ->  GPU BEOL  ->  HB microbumps
    ->  N x ( NVDRAM-tier BEOL  ->  NVDRAM-tier FEOL )
    (PACT then appends heat spreader + heat sink automatically)

Each device tier is split into FEOL (front-end transistors) and BEOL (back-end
metal) sublayers. The GPU die is a full GPU_SIDE square; the NVDRAM memory macro
is a smaller MEM_SIDE square centred on it, with dielectric FILLER occupying the
surrounding frame in every memory and microbump layer.

This script writes:
  * 6 floorplan CSVs  (shared 9-block geometry; labels differ per layer type)
  * 6 ptrace CSVs     (power maps; GPU TDP distributed, NVDRAM idle = 0)
  * stacked_n{N}_lcf.csv  (the layer configuration file)
and prints a stack summary (thickness + analytic thermal resistance per layer).

The material properties and solver settings live in the static files
`experiment.config` and `modelParams.config`.

Usage:
    python generate_stacked.py [--n N] [--gpu-tdp W] [--gpu-side M] [--mem-side M]

    --n N          Number of NVDRAM tiers (default: 4); each tier = BEOL + FEOL.
    --gpu-tdp W    GPU active power in watts (default: 75).
    --gpu-side M   GPU die edge length in metres (default: 0.017).
    --mem-side M   NVDRAM memory macro edge length in metres (default: 0.0085).
"""

import argparse

# Defaults
GPU_SIDE  = 0.017     # m  (17 mm GPU base die)
MEM_SIDE  = 0.0085    # m  (8.5 mm NVDRAM macro, centred on the GPU die)
GPU_TDP   = 75.0      # W
IDLE_FRAC = 0.15      # GPU idle power as a fraction of TDP (volatile leakage)
EPS       = 1e-6      # tiny placeholder power for passive blocks (W)

# Per-layer-type definition:  (flp_file, ptrace_file, thickness_m, k_W_mK, summary_label)
# k is the *active/centre* material conductivity, used only for the printed R_th summary.
LAYER_DEFS = {
    "gpu_substrate": ("gpu_substrate_flp.csv", "gpu_substrate_ptrace.csv", 1.0e-4, 130.0, "GPU substrate/TSV (Si)"),
    "gpu_feol":      ("gpu_feol_flp.csv",      "gpu_feol_ptrace.csv",      1.0e-6, 110.0, "GPU FEOL/TSV (active Si)"),
    "gpu_beol":      ("gpu_beol_flp.csv",      "gpu_beol_ptrace.csv",      5.0e-6,  12.0, "GPU BEOL (Cu/low-k)"),
    "hb_ubump":      ("hb_ubump_flp.csv",      "hb_ubump_ptrace.csv",      5.0e-6,  20.0, "HB microbumps (Cu/underfill)"),
    "nvdram_beol":   ("nvdram_beol_flp.csv",   "nvdram_beol_ptrace.csv",   1.0e-6,  12.0, "NVDRAM BEOL (Cu interconnect)"),
    "nvdram_feol":   ("nvdram_feol_flp.csv",   "nvdram_feol_ptrace.csv",   1.0e-6,  14.0, "NVDRAM FEOL (1T1C/HZO)"),
}

# Block-name order shared by every floorplan and ptrace.
CENTRE_BLOCKS = ["MemArray", "RowDecoder_L", "RowDecoder_R", "SenseAmp_B", "SenseAmp_T"]
FRAME_BLOCKS  = ["Frame_B", "Frame_T", "Frame_L", "Frame_R"]
ALL_BLOCKS    = CENTRE_BLOCKS + FRAME_BLOCKS


def geometry(gpu_side, mem_side):
    """Return {block_name: (X, Y, Length, Width)} tiling the gpu_side square.

    The centre mem_side x mem_side macro uses the standard 5-block memory
    layout; four FILLER frame blocks tile the surrounding border.
    """
    off    = (gpu_side - mem_side) / 2.0     # filler border width on each side
    margin = mem_side * (1.0 / 8.5)          # decoder / sense-amp strip width
    array  = mem_side - 2.0 * margin
    geo = {
        # centre macro (offset onto the GPU die)
        "MemArray":     (off + margin,            off + margin,            array,    array),
        "RowDecoder_L": (off + 0.0,               off + margin,            margin,   array),
        "RowDecoder_R": (off + mem_side - margin, off + margin,            margin,   array),
        "SenseAmp_B":   (off + 0.0,               off + 0.0,               mem_side, margin),
        "SenseAmp_T":   (off + 0.0,               off + mem_side - margin, mem_side, margin),
        # filler frame
        "Frame_B": (0.0,            0.0,            gpu_side, off),
        "Frame_T": (0.0,            off + mem_side, gpu_side, off),
        "Frame_L": (0.0,            off,            off,      mem_side),
        "Frame_R": (off + mem_side, off,            off,      mem_side),
    }
    return geo


def write_flp(path, geo, centre_label, frame_label):
    """Write a floorplan: centre blocks get centre_label, frame blocks frame_label."""
    lines = ["UnitName,X,Y,Length (m),Width (m),ConfigFile,Label"]
    for name in ALL_BLOCKS:
        x, y, l, w = geo[name]
        label = centre_label if name in CENTRE_BLOCKS else frame_label
        lines.append(f"{name},{x:g},{y:g},{l:g},{w:g},,{label}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def write_ptrace(path, powers):
    """Write a 3-column (Idle, Read, Write) ptrace. powers maps block -> (idle, read, write)."""
    lines = ["UnitName,Power,Power1,Power2"]
    for name in ALL_BLOCKS:
        i, r, w = powers[name]
        lines.append(f"{name},{i:g},{r:g},{w:g}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def passive_powers():
    """Every block dissipates a negligible EPS in all scenarios."""
    return {name: (EPS, EPS, EPS) for name in ALL_BLOCKS}


def gpu_feol_powers(geo, gpu_side, tdp):
    """Distribute the GPU TDP uniformly by area over the full die (idle = IDLE_FRAC*TDP)."""
    density_active = tdp / (gpu_side * gpu_side)
    density_idle   = IDLE_FRAC * density_active
    powers = {}
    for name in ALL_BLOCKS:
        _, _, l, w = geo[name]
        area = l * w
        powers[name] = (density_idle * area, density_active * area, density_active * area)
    return powers


def nvdram_powers(centre_map):
    """NVDRAM tier: idle = 0 (non-volatile, power-gated); centre blocks carry the
    given (read, write); FILLER frame blocks are passive (EPS)."""
    powers = {}
    for name in CENTRE_BLOCKS:
        r, w = centre_map[name]
        powers[name] = (0.0, r, w)
    for name in FRAME_BLOCKS:
        powers[name] = (0.0, EPS, EPS)
    return powers


# NVDRAM ferroelectric power per tier (W). idle = 0; write > read because HZO
# polarisation switching + write-verify dominates; read is raised by the
# destructive-read + write-back of a 1T1C ferroelectric cell.
NV_FEOL_RW = {
    "MemArray":     (0.40, 0.80),
    "RowDecoder_L": (0.08, 0.15),
    "RowDecoder_R": (0.08, 0.15),
    "SenseAmp_B":   (0.07, 0.10),
    "SenseAmp_T":   (0.07, 0.10),
}  # centre totals: read 0.70 W, write 1.30 W
NV_BEOL_RW = {
    "MemArray":     (0.080, 0.16),
    "RowDecoder_L": (0.020, 0.04),
    "RowDecoder_R": (0.020, 0.04),
    "SenseAmp_B":   (0.015, 0.03),
    "SenseAmp_T":   (0.015, 0.03),
}  # centre totals: read 0.15 W, write 0.30 W


def write_inputs(geo, gpu_side, tdp):
    """Write all 6 floorplans and 6 ptraces."""
    # Floorplans: (centre_label, frame_label)
    write_flp("gpu_substrate_flp.csv", geo, "GPU_Si",   "GPU_Si")
    write_flp("gpu_feol_flp.csv",      geo, "GPU_FEOL", "GPU_FEOL")
    write_flp("gpu_beol_flp.csv",      geo, "GPU_BEOL", "GPU_BEOL")
    write_flp("hb_ubump_flp.csv",      geo, "HB_uBump", "FILLER")
    write_flp("nvdram_beol_flp.csv",   geo, "NV_BEOL",  "FILLER")
    write_flp("nvdram_feol_flp.csv",   geo, "NV_FEOL",  "FILLER")

    # Ptraces
    write_ptrace("gpu_substrate_ptrace.csv", passive_powers())
    write_ptrace("gpu_feol_ptrace.csv",      gpu_feol_powers(geo, gpu_side, tdp))
    write_ptrace("gpu_beol_ptrace.csv",      passive_powers())
    write_ptrace("hb_ubump_ptrace.csv",      passive_powers())
    write_ptrace("nvdram_beol_ptrace.csv",   nvdram_powers(NV_BEOL_RW))
    write_ptrace("nvdram_feol_ptrace.csv",   nvdram_powers(NV_FEOL_RW))


def build_layers(n):
    """Ordered list (bottom->top) of (name, LAYER_DEFS-entry)."""
    layers = [
        ("gpu_substrate", LAYER_DEFS["gpu_substrate"]),
        ("gpu_feol",      LAYER_DEFS["gpu_feol"]),
        ("gpu_beol",      LAYER_DEFS["gpu_beol"]),
        ("hb_ubump",      LAYER_DEFS["hb_ubump"]),
    ]
    for _ in range(n):
        layers.append(("nvdram_beol", LAYER_DEFS["nvdram_beol"]))
        layers.append(("nvdram_feol", LAYER_DEFS["nvdram_feol"]))
    return layers


def write_lcf(layers, path):
    lines = ["Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow"]
    for idx, (_, (flp, ptrace, thickness, _, _)) in enumerate(layers):
        lines.append(f"{idx},{flp},{thickness:.2e},{ptrace},True")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def print_summary(layers, gpu_side, title):
    area = gpu_side * gpu_side
    total_r = total_t = 0.0
    print(f"\n{'-'*72}")
    print(f"  {title}")
    print(f"{'-'*72}")
    print(f"  {'Layer':<5} {'Material':<32} {'Thick (um)':>10} {'R_th (mK/W)':>12}")
    print(f"  {'-'*5} {'-'*32} {'-'*10} {'-'*12}")
    for idx, (_, (_, _, thickness, k, label)) in enumerate(layers):
        rv = thickness / (k * area) * 1e3
        total_r += rv
        total_t += thickness * 1e6
        print(f"  {idx:<5} {label:<32} {thickness*1e6:>10.1f} {rv:>12.3f}")
    print(f"  {'-'*5} {'-'*32} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL':<5} {'(heat spreader + sink added by PACT)':<32} {total_t:>10.1f} {total_r:>12.3f}")
    print(f"{'-'*72}\n")


def main():
    p = argparse.ArgumentParser(description="Generate PACT inputs for an NVDRAM-on-GPU 3D stack.")
    p.add_argument("--n",        type=int,   default=4,        help="Number of NVDRAM tiers (default: 4)")
    p.add_argument("--gpu-tdp",  type=float, default=GPU_TDP,  help="GPU active power, W (default: 75)")
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE, help="GPU die edge, m (default: 0.017)")
    p.add_argument("--mem-side", type=float, default=MEM_SIDE, help="NVDRAM macro edge, m (default: 0.0085)")
    args = p.parse_args()

    if args.mem_side >= args.gpu_side:
        raise SystemExit("--mem-side must be smaller than --gpu-side (memory macro sits inside the GPU die).")

    geo = geometry(args.gpu_side, args.mem_side)
    print(f"\nGenerating NVDRAM-on-GPU stack: n={args.n} tiers, "
          f"GPU {args.gpu_side*1e3:g} mm @ {args.gpu_tdp:g} W, "
          f"memory macro {args.mem_side*1e3:g} mm centred.\n")
    write_inputs(geo, args.gpu_side, args.gpu_tdp)

    layers = build_layers(args.n)
    lcf_path = f"stacked_n{args.n}_lcf.csv"
    write_lcf(layers, lcf_path)
    print_summary(layers, args.gpu_side,
                  f"NVDRAM-on-GPU 3D stack (N={args.n} tiers, GPU {args.gpu_tdp:g} W)")


if __name__ == "__main__":
    main()
