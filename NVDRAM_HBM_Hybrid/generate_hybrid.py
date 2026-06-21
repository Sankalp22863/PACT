"""
Hybrid NVDRAM-bottom / DRAM-top 3D-Integration Stack Generator
==============================================================
A "future 3D integration" where BOTH memory technologies are stacked directly
on top of a GPU compute die and cooled by a top-side liquid lid:

    * the NON-VOLATILE NVDRAM tiers sit at the BOTTOM of the memory stack
      (nearest the GPU), and
    * the normal (HBM) DRAM tiers sit on TOP (nearest the cold-plate lid).

Stack, bottom (laminate/package side, layer 0) -> top (lid / cold-plate side):

    GPU substrate/TSV -> GPU FEOL (heat source) -> GPU BEOL
      -> uBump / hybrid-bond interface
      -> NV_TIERS x  NVDRAM die  (each separated from the next by a DAF bond)
      -> DRAM_TIERS x DRAM die   (each separated from the next by a DAF bond)
    (PACT then appends the NoPackage convective lid on top automatically)

Lateral layout reuses the proven DRAM_3D_Stacked geometry: four memory stacks
in a 2x2 arrangement; the cross-shaped gap between them and the outer frame are
filled with high-conductivity "thermal silicon" (140 W/m.K). A vertical
cross-section through the die centre therefore shows  memory | thermal-Si |
memory  -- i.e. memory on both sides with a central thermal-silicon column.

PACT cools only from the top lid; the bottom of layer 0 is adiabatic, so heat
is forced *up* through the whole memory stack to the lid.

Usage:
    python3 generate_hybrid.py [--nv-tiers N] [--dram-tiers N] [--gpu-power W]
                               [--nv-per-stack W] [--dram-per-stack W]
                               [--gpu-side M] [--mem-side M] [--grid N]
"""

import argparse

# ---- defaults ----
GPU_SIDE       = 0.024     # m  (24 mm GPU base die)
MEM_SIDE       = 0.010     # m  (10 mm memory-stack footprint)
GPU_POWER      = 414.0     # W  (GPU active heat source)
NV_PER_STACK   = 8.0       # W  per NVDRAM stack (low: non-volatile, power-gated)
DRAM_PER_STACK = 40.0      # W  per 12-Hi HBM-style DRAM stack
NV_TIERS       = 4         # NVDRAM dies per stack (bottom block)
DRAM_TIERS     = 8         # DRAM dies per stack (top block)
GRID           = 48
EPS            = 1e-6      # passive-block placeholder power (W)

# Layer thicknesses (m)
T_GPU_SUB  = 100e-6
T_GPU_FEOL = 2e-6
T_GPU_BEOL = 8e-6
T_UBUMP    = 15e-6
T_NV       = 40e-6     # per NVDRAM die (incl. its thin BEOL)
T_DRAM     = 50e-6     # per thinned DRAM die (incl. its thin BEOL)
T_DAF      = 5e-6      # die-attach film between stacked dies (the bottleneck)


def gpu_block(gpu_side, label):
    """Single full-die block covering the whole GPU footprint."""
    return [(f"{label}", 0.0, 0.0, gpu_side, gpu_side, label)]


def mem_tier_blocks(gpu_side, mem_side, macro_label):
    """4 memory stacks in 2x2 + thermal-silicon cross/frame filling the rest.

    macro_label selects what the four stack footprints are made of:
      "NV_FEOL"  for an NVDRAM-die layer,
      "DRAM_Si"  for an HBM DRAM-die layer, or
      "DAF"      for an inter-die die-attach-film layer.
    The cross/frame is always THERMAL_SI (continuous high-k escape path), so the
    bottleneck lives only inside the memory stacks.
    """
    g = gpu_side
    h = mem_side
    gap = g - 2 * h                      # margin | h | central | h | margin
    margin = gap / 4.0                   # outer frame width
    central = gap / 2.0                  # central cross width
    a0 = margin
    a1 = margin + h
    a2 = margin + h + central
    a3 = margin + h + central + h

    blocks = []
    # --- 4 memory stacks ---
    blocks.append(("MEM_BL", a0, a0, h, h, macro_label))
    blocks.append(("MEM_BR", a2, a0, h, h, macro_label))
    blocks.append(("MEM_TL", a0, a2, h, h, macro_label))
    blocks.append(("MEM_TR", a2, a2, h, h, macro_label))
    # --- outer frame (thermal silicon) ---
    blocks.append(("Frame_B", 0.0, 0.0, g, margin, "THERMAL_SI"))
    blocks.append(("Frame_T", 0.0, a3, g, margin, "THERMAL_SI"))
    blocks.append(("Frame_L", 0.0, margin, margin, g - 2 * margin, "THERMAL_SI"))
    blocks.append(("Frame_R", a3, margin, margin, g - 2 * margin, "THERMAL_SI"))
    # --- central cross (thermal silicon) ---
    blocks.append(("Cross_V", a1, margin, central, g - 2 * margin, "THERMAL_SI"))
    blocks.append(("Cross_HL", a0, a1, h, central, "THERMAL_SI"))
    blocks.append(("Cross_HR", a2, a1, h, central, "THERMAL_SI"))
    return blocks


def write_flp(path, blocks):
    lines = ["UnitName,X,Y,Length (m),Width (m),ConfigFile,Label"]
    for name, x, y, l, w, label in blocks:
        lines.append(f"{name},{x:g},{y:g},{l:g},{w:g},,{label}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def write_ptrace(path, names, power_map):
    """3-column (steady) ptrace; all columns equal so I_avg == the active power."""
    lines = ["UnitName,Power,Power1,Power2"]
    for n in names:
        p = power_map.get(n, EPS)
        lines.append(f"{n},{p:g},{p:g},{p:g}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def write_lcf(layers, path):
    lines = ["Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow"]
    for idx, (flp, thickness, ptrace) in enumerate(layers):
        lines.append(f"{idx},{flp},{thickness:.2e},{ptrace},True")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def main():
    p = argparse.ArgumentParser(
        description="Generate PACT inputs for a hybrid NVDRAM-bottom / DRAM-top 3D stack.")
    p.add_argument("--nv-tiers", type=int, default=NV_TIERS)
    p.add_argument("--dram-tiers", type=int, default=DRAM_TIERS)
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--nv-per-stack", type=float, default=NV_PER_STACK)
    p.add_argument("--dram-per-stack", type=float, default=DRAM_PER_STACK)
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE)
    p.add_argument("--mem-side", type=float, default=MEM_SIDE)
    p.add_argument("--grid", type=int, default=GRID)
    args = p.parse_args()

    if args.mem_side * 2 >= args.gpu_side:
        raise SystemExit("Need 2*mem-side < gpu-side so the 2x2 memory array fits with margins.")

    print(f"\nGenerating hybrid NVDRAM-bottom / DRAM-top stack: "
          f"GPU {args.gpu_side*1e3:g} mm @ {args.gpu_power:g} W, "
          f"{args.nv_tiers} NVDRAM tiers @ {args.nv_per_stack:g} W/stack + "
          f"{args.dram_tiers} DRAM tiers @ {args.dram_per_stack:g} W/stack "
          f"(4 stacks, 2x2).\n")

    # ---- GPU layer floorplans + ptraces ----
    write_flp("gpu_substrate_flp.csv", gpu_block(args.gpu_side, "GPU_Si"))
    write_flp("gpu_feol_flp.csv",      gpu_block(args.gpu_side, "GPU_FEOL"))
    write_flp("gpu_beol_flp.csv",      gpu_block(args.gpu_side, "GPU_BEOL"))
    write_flp("ubump_flp.csv",         gpu_block(args.gpu_side, "UBUMP"))
    write_ptrace("gpu_substrate_ptrace.csv", ["GPU_Si"], {})
    write_ptrace("gpu_feol_ptrace.csv",      ["GPU_FEOL"], {"GPU_FEOL": args.gpu_power})
    write_ptrace("gpu_beol_ptrace.csv",      ["GPU_BEOL"], {})
    write_ptrace("ubump_ptrace.csv",         ["UBUMP"], {})

    # ---- memory-tier floorplans (NVDRAM die, DRAM die, DAF bond) ----
    nv_blocks   = mem_tier_blocks(args.gpu_side, args.mem_side, "NV_FEOL")
    dram_blocks = mem_tier_blocks(args.gpu_side, args.mem_side, "DRAM_Si")
    daf_blocks  = mem_tier_blocks(args.gpu_side, args.mem_side, "DAF")
    names = [b[0] for b in nv_blocks]  # identical block names across tier types
    write_flp("nv_tier_flp.csv",   nv_blocks)
    write_flp("dram_tier_flp.csv", dram_blocks)
    write_flp("daf_flp.csv",       daf_blocks)

    nv_per_die   = args.nv_per_stack / args.nv_tiers
    dram_per_die = args.dram_per_stack / args.dram_tiers
    write_ptrace("nv_tier_ptrace.csv",   names,
                 {n: nv_per_die for n in names if n.startswith("MEM_")})
    write_ptrace("dram_tier_ptrace.csv", names,
                 {n: dram_per_die for n in names if n.startswith("MEM_")})
    write_ptrace("daf_ptrace.csv",       names, {})

    # ---- assemble the layer stack ----
    layers = [
        ("gpu_substrate_flp.csv", T_GPU_SUB,  "gpu_substrate_ptrace.csv"),
        ("gpu_feol_flp.csv",      T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("gpu_beol_flp.csv",      T_GPU_BEOL, "gpu_beol_ptrace.csv"),
        ("ubump_flp.csv",         T_UBUMP,    "ubump_ptrace.csv"),
    ]
    # NVDRAM block (bottom of memory): die, DAF, die, DAF, ...
    for t in range(args.nv_tiers):
        layers.append(("nv_tier_flp.csv", T_NV, "nv_tier_ptrace.csv"))
        layers.append(("daf_flp.csv", T_DAF, "daf_ptrace.csv"))  # incl. NV->DRAM bond
    # DRAM block (top of memory): die, DAF, die, ... (no trailing DAF after the top die)
    for t in range(args.dram_tiers):
        layers.append(("dram_tier_flp.csv", T_DRAM, "dram_tier_ptrace.csv"))
        if t < args.dram_tiers - 1:
            layers.append(("daf_flp.csv", T_DAF, "daf_ptrace.csv"))

    lcf_path = "hybrid_lcf.csv"
    write_lcf(layers, lcf_path)

    total_mem = 4 * (args.nv_per_stack + args.dram_per_stack)
    print(f"\n  Layers: {len(layers)} device layers"
          f" (3 GPU + 1 interface + {args.nv_tiers} NVDRAM + {args.dram_tiers} DRAM + DAF bonds)"
          f"  + 1 NoPackage lid (added by PACT)")
    print(f"  Power : GPU {args.gpu_power:g} W + memory {total_mem:g} W"
          f" = {args.gpu_power + total_mem:g} W total")
    print(f"  Grid  : {args.grid}x{args.grid} on {args.gpu_side*1e3:g} mm die"
          f"  ({args.gpu_side/args.grid*1e3:.3f} mm/cell)")
    print(f"  LCF   : {lcf_path}\n")


if __name__ == "__main__":
    main()
