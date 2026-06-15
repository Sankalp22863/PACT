"""
NVDRAM / DRAM / Hybrid Stack LCF Generator
===========================================
Generates PACT layer configuration files (LCF) for three stack topologies,
parameterised by N (number of memory dies/layers per chip type):

  nvdram   — CMOS + N × 1T1C PolySi layers  (monolithic 3D)
  dram     — CMOS + N × (FusionBond + DRAM die)  (wafer-bonded 3D)
  hybrid   — CMOS + N × 1T1C + N × (FusionBond + DRAM die)  (combined stack)

Usage:
    python generate_stack.py [--n N] [--type nvdram|dram|hybrid|all] [--out-prefix PREFIX]

    --n N            Number of memory layers per type (default: 4)
    --type TYPE      Which LCF(s) to generate (default: all)
    --out-prefix P   Filename prefix (default: derived from type and N)

Examples:
    python generate_stack.py --n 4                  # all three, N=4
    python generate_stack.py --n 8 --type hybrid    # hybrid N=8 only
    python generate_stack.py --n 2 --type dram      # DRAM-only N=2
"""

import argparse
import os

# ── physical constants ────────────────────────────────────────────────────────
CHIP_AREA     = 0.0085 * 0.0085   # m²  (8.5 mm × 8.5 mm)

LAYER_DEFS = {
    # name           : (floorplan_file,       ptrace_file,           thickness_m, k_W_mK,  label)
    "cmos"           : ("cmos_flp.csv",        "nvdram_cmos_ptrace.csv", 3.0e-4,  130.0,  "Si (CMOS, 300 µm)"),
    "cmos_dram"      : ("cmos_flp.csv",        "dram_cmos_ptrace.csv",   3.0e-4,  130.0,  "Si (CMOS+refresh, 300 µm)"),
    "nvdram_mem"     : ("nvdram_mem_flp.csv",  "nvdram_mem_ptrace.csv",  2.0e-6,   14.0,  "PolySi1T1C (1T1C, 2 µm)"),
    "dram_bond"      : ("dram_bond_flp.csv",   "dram_bond_ptrace.csv",   2.0e-6,    1.4,  "FusionBond (SiO₂, 2 µm)"),
    "dram_die"       : ("dram_die_flp.csv",    "dram_die_ptrace.csv",    5.0e-5,  130.0,  "Si (DRAM die, 50 µm)"),
}


def r_th(thickness, k):
    """Layer thermal resistance in mK/W."""
    return thickness / (k * CHIP_AREA) * 1e3


def build_nvdram_layers(n):
    """CMOS + N × 1T1C."""
    layers = [("cmos", LAYER_DEFS["cmos"])]
    for _ in range(n):
        layers.append(("nvdram_mem", LAYER_DEFS["nvdram_mem"]))
    return layers


def build_dram_layers(n):
    """CMOS + N × (bond + die)."""
    layers = [("cmos_dram", LAYER_DEFS["cmos_dram"])]
    for _ in range(n):
        layers.append(("dram_bond", LAYER_DEFS["dram_bond"]))
        layers.append(("dram_die",  LAYER_DEFS["dram_die"]))
    return layers


def build_hybrid_layers(n):
    """CMOS + N × 1T1C + bond + N × (bond + die).
    The interface between the NVDRAM section and the first DRAM die
    needs a fusion bond (the DRAM die is wafer-bonded onto the NVDRAM stack).
    """
    layers = [("cmos", LAYER_DEFS["cmos"])]
    for _ in range(n):
        layers.append(("nvdram_mem", LAYER_DEFS["nvdram_mem"]))
    # bond between NVDRAM stack and first DRAM die
    for _ in range(n):
        layers.append(("dram_bond", LAYER_DEFS["dram_bond"]))
        layers.append(("dram_die",  LAYER_DEFS["dram_die"]))
    return layers


def write_lcf(layers, out_path):
    lines = ["Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow"]
    for idx, (_, (flp, ptrace, thickness, _, _)) in enumerate(layers):
        lines.append(f"{idx},{flp},{thickness:.2e},{ptrace},True")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {out_path}")


def print_stack_summary(layers, title):
    total_r   = 0.0
    total_t   = 0.0
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"{'─'*68}")
    print(f"  {'Layer':<5} {'Material':<32} {'Thick (µm)':>10} {'R_th (mK/W)':>12}")
    print(f"  {'─'*5} {'─'*32} {'─'*10} {'─'*12}")
    for idx, (_, (_, _, thickness, k, label)) in enumerate(layers):
        rv    = r_th(thickness, k)
        total_r += rv
        total_t += thickness * 1e6   # µm
        print(f"  {idx:<5} {label:<32} {thickness*1e6:>10.1f} {rv:>12.2f}")
    print(f"  {'─'*5} {'─'*32} {'─'*10} {'─'*12}")
    print(f"  {'TOTAL':<5} {'':<32} {total_t:>10.1f} {total_r:>12.2f}")
    print(f"{'─'*68}\n")


def generate(stack_type, n, out_prefix=None):
    builders = {
        "nvdram": (build_nvdram_layers, "NVDRAM monolithic 3D"),
        "dram":   (build_dram_layers,   "3D-bonded DRAM"),
        "hybrid": (build_hybrid_layers, f"Hybrid: {n}×NVDRAM + {n}×DRAM"),
    }
    build_fn, title = builders[stack_type]
    layers = build_fn(n)

    if out_prefix is None:
        out_prefix = f"{stack_type}_n{n}"
    out_path = f"{out_prefix}_lcf.csv"

    write_lcf(layers, out_path)
    print_stack_summary(layers, f"{title}  (N={n})")
    return layers


def main():
    parser = argparse.ArgumentParser(description="Generate PACT LCF files for NVDRAM/DRAM/Hybrid stacks.")
    parser.add_argument("--n",          type=int,   default=4,    help="Number of memory layers per type (default: 4)")
    parser.add_argument("--type",       type=str,   default="all", choices=["nvdram", "dram", "hybrid", "all"])
    parser.add_argument("--out-prefix", type=str,   default=None,  help="Output filename prefix (overrides default)")
    args = parser.parse_args()

    types = ["nvdram", "dram", "hybrid"] if args.type == "all" else [args.type]
    for t in types:
        prefix = args.out_prefix if (args.out_prefix and len(types) == 1) else None
        generate(t, args.n, prefix)


if __name__ == "__main__":
    main()
