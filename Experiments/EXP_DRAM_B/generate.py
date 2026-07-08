"""
3D HBM-on-GPU Stack Generator  (paper BASELINE = "3D thermal penalty", no thermal silicon)
==========================================================================================
Same detailed HBM-on-GPU stack as EXP_DRAM_R, but this variant reproduces the
paper's *un-optimized* baseline -- the leftmost "3D thermal penalty" bar of the
IEDM STCO staircase (GPU peak 141.7 C in "Breaking Thermal Bottleneck in 3D
HBM-on-GPU Integration via System-Technology Co-Optimization", imec, IEDM 2025).

The ONLY difference from EXP_DRAM_R is the cross/frame filler material:

  * EXP_DRAM_R (optimized) : cross/frame = THERMAL_SI  (k = 140 W/m-K)
                             -> high-k vertical/lateral heat-escape paths
                             -> "thermal silicon optimization" already applied.
  * EXP_DRAM_B (baseline)  : cross/frame = MOLD         (k = 0.8 W/m-K)
                             -> standard epoxy mold compound between/around the
                                HBM stacks; no thermal-silicon escape path, so
                                heat is forced up through the low-k DRAM BEOL /
                                hybrid-bond bottleneck -> the "thermal penalty".

The paper does not publish the exact baseline gap-fill conductivity, so MOLD uses
the industry-standard epoxy mold compound value (k ~ 0.8 W/m-K); tune it in
experiment.config or via --mold-k.

Stack, bottom (adiabatic package side, layer 0) -> top (lid / cold-plate side):

    BSPDN -> GPU FEOL (heat source, 414 W) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> HBM Base BEOL -> HBM Base Si (Base Die)
      -> 12 x [ Hybrid Bonding -> DRAM Die BEOL -> DRAM Die Si ]   (top die = 169 um)
      -> TIM -> Lid
    (PACT then applies the NoPackage convective HTC boundary on top of the Lid)

Usage:
    python generate.py [--gpu-power W] [--hbm-per-stack W] [--tiers N]
                       [--gpu-side M] [--hbm-side M] [--grid N] [--no-base-die]
                       [--optimized]     # use THERMAL_SI instead of MOLD (= EXP_DRAM_R)
"""

import argparse

# ---- defaults (paper-representative) ----
GPU_SIDE   = 0.024     # m  (24 mm GPU base die)
HBM_SIDE   = 0.010     # m  (10 mm HBM stack footprint)
GPU_POWER  = 414.0     # W  (nominal 1.0f AI workload)
HBM_PER_STACK = 40.0   # W  per 12-Hi stack
TIERS      = 12        # DRAM dies per stack
GRID       = 48
EPS        = 1e-6      # passive-block placeholder power (W)

# ---- cross/frame filler: BASELINE uses mold (no thermal silicon) ----
FILLER_BASELINE  = "MOLD"        # epoxy mold compound, k ~ 0.8 W/m-K  (paper baseline)
FILLER_OPTIMIZED = "THERMAL_SI"  # high-k dummy silicon, k = 140       (= EXP_DRAM_R)

# ---- layer thicknesses (m), from the paper cross-section table ----
T_BSPDN      = 1.715e-6
T_GPU_FEOL   = 0.15e-6
T_BEOL_MXY   = 1.4e-6
T_OXIDE      = 1.0e-6
T_UBUMP      = 40e-6      # GPU-HBM uBump
T_BASE_BEOL  = 5e-6       # HBM Base BEOL
T_BASE_SI    = 50e-6      # HBM Base Si (Base Die)
T_HYBOND     = 2e-6       # Hybrid Bonding
T_DRAM_BEOL  = 3e-6       # DRAM Die BEOL
T_DRAM_SI    = 50e-6      # inner DRAM Die Si (thinned)
T_DRAM_TOP   = 169e-6     # top DRAM Die Si
T_TIM        = 200e-6
T_LID        = 3000e-6


def gpu_block(gpu_side, label):
    """Single full-die block covering the whole footprint."""
    return [(f"{label}", 0.0, 0.0, gpu_side, gpu_side, label)]


def hbm_tier_blocks(gpu_side, hbm_side, macro_label, filler_label):
    """4 HBM stacks in 2x2 + cross/frame filler filling the rest.

    The four stack footprints carry `macro_label`; the cross/frame is
    `filler_label` -- THERMAL_SI (k=140, optimized) or MOLD (k=0.8, baseline).
    In the baseline the filler is a poor vertical conductor, so the inter-stack
    region no longer provides a heat-escape path. Returns (name, X, Y, L, W,
    Label) blocks tiling the die with no gaps/overlaps.
    """
    g, h = gpu_side, hbm_side
    gap = g - 2 * h
    margin = gap / 4.0
    central = gap / 2.0
    a0 = margin
    a1 = margin + h
    a2 = margin + h + central
    a3 = margin + h + central + h

    blocks = []
    blocks.append(("HBM_BL", a0, a0, h, h, macro_label))
    blocks.append(("HBM_BR", a2, a0, h, h, macro_label))
    blocks.append(("HBM_TL", a0, a2, h, h, macro_label))
    blocks.append(("HBM_TR", a2, a2, h, h, macro_label))
    blocks.append(("Frame_B", 0.0, 0.0, g, margin, filler_label))
    blocks.append(("Frame_T", 0.0, a3, g, margin, filler_label))
    blocks.append(("Frame_L", 0.0, margin, margin, g - 2 * margin, filler_label))
    blocks.append(("Frame_R", a3, margin, margin, g - 2 * margin, filler_label))
    blocks.append(("Cross_V", a1, margin, central, g - 2 * margin, filler_label))
    blocks.append(("Cross_HL", a0, a1, h, central, filler_label))
    blocks.append(("Cross_HR", a2, a1, h, central, filler_label))
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
        lines.append(f"{idx},{flp},{thickness:.3e},{ptrace},True")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Written: {path}")


def main():
    p = argparse.ArgumentParser(description="Generate PACT inputs for the BASELINE 3D HBM-on-GPU stack (no thermal silicon).")
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--hbm-per-stack", type=float, default=HBM_PER_STACK)
    p.add_argument("--tiers", type=int, default=TIERS)
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE)
    p.add_argument("--hbm-side", type=float, default=HBM_SIDE)
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--no-base-die", action="store_true",
                   help="Remove the HBM base die (the paper's 'base die removal' study)")
    p.add_argument("--optimized", action="store_true",
                   help="Use THERMAL_SI (k=140) cross/frame instead of MOLD -> reproduces EXP_DRAM_R")
    args = p.parse_args()

    if args.hbm_side * 2 >= args.gpu_side:
        raise SystemExit("Need 2*hbm-side < gpu-side so the 2x2 HBM array fits with margins.")

    filler = FILLER_OPTIMIZED if args.optimized else FILLER_BASELINE

    print(f"\nGenerating {'OPTIMIZED' if args.optimized else 'BASELINE'} 3D HBM-on-GPU stack: "
          f"GPU {args.gpu_side*1e3:g} mm @ {args.gpu_power:g} W, "
          f"4 HBM stacks @ {args.hbm_per_stack:g} W ({args.tiers}-Hi"
          f"{'' if not args.no_base_die else ', base die removed'});"
          f" cross/frame filler = {filler}.\n")

    # ---- full-die GPU + interface + package floorplans ----
    write_flp("bspdn_flp.csv",    gpu_block(args.gpu_side, "BSPDN"))
    write_flp("gpu_feol_flp.csv", gpu_block(args.gpu_side, "GPU_FEOL"))
    write_flp("beol_mxy_flp.csv", gpu_block(args.gpu_side, "BEOL_MXY"))
    write_flp("oxide_flp.csv",    gpu_block(args.gpu_side, "OXIDE"))
    write_flp("ubump_flp.csv",    gpu_block(args.gpu_side, "GPU_HBM_UBUMP"))
    write_flp("tim_flp.csv",      gpu_block(args.gpu_side, "TIM"))
    write_flp("lid_flp.csv",      gpu_block(args.gpu_side, "LID"))

    write_ptrace("bspdn_ptrace.csv",    ["BSPDN"], {})
    write_ptrace("gpu_feol_ptrace.csv", ["GPU_FEOL"], {"GPU_FEOL": args.gpu_power})
    write_ptrace("beol_mxy_ptrace.csv", ["BEOL_MXY"], {})
    write_ptrace("oxide_ptrace.csv",    ["OXIDE"], {})
    write_ptrace("ubump_ptrace.csv",    ["GPU_HBM_UBUMP"], {})
    write_ptrace("tim_ptrace.csv",      ["TIM"], {})
    write_ptrace("lid_ptrace.csv",      ["LID"], {})

    # ---- HBM-stack sublayer floorplans (2x2 macros + cross/frame filler) ----
    base_beol = hbm_tier_blocks(args.gpu_side, args.hbm_side, "HBM_BASE_BEOL", filler)
    base_si   = hbm_tier_blocks(args.gpu_side, args.hbm_side, "HBM_BASE_SI", filler)
    hybond    = hbm_tier_blocks(args.gpu_side, args.hbm_side, "HYBRID_BOND", filler)
    dram_beol = hbm_tier_blocks(args.gpu_side, args.hbm_side, "DRAM_BEOL", filler)
    dram_si   = hbm_tier_blocks(args.gpu_side, args.hbm_side, "DRAM_SI", filler)
    names = [b[0] for b in dram_si]
    write_flp("hbm_base_beol_flp.csv", base_beol)
    write_flp("hbm_base_si_flp.csv",   base_si)
    write_flp("hybrid_bond_flp.csv",   hybond)
    write_flp("dram_beol_flp.csv",     dram_beol)
    write_flp("dram_tier_flp.csv",     dram_si)   # DRAM Die Si (the "tier")

    per_die = args.hbm_per_stack / args.tiers
    write_ptrace("dram_tier_ptrace.csv", names,
                 {n: per_die for n in names if n.startswith("HBM_")})
    # passive HBM-stack sublayers (one shared passive ptrace)
    write_ptrace("hbm_passive_ptrace.csv", names, {})

    # ---- assemble the layer stack (bottom -> top) ----
    layers = [
        ("bspdn_flp.csv",    T_BSPDN,    "bspdn_ptrace.csv"),
        ("gpu_feol_flp.csv", T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("beol_mxy_flp.csv", T_BEOL_MXY, "beol_mxy_ptrace.csv"),
        ("oxide_flp.csv",    T_OXIDE,    "oxide_ptrace.csv"),
        ("ubump_flp.csv",    T_UBUMP,    "ubump_ptrace.csv"),
    ]
    if not args.no_base_die:
        layers.append(("hbm_base_beol_flp.csv", T_BASE_BEOL, "hbm_passive_ptrace.csv"))
        layers.append(("hbm_base_si_flp.csv",   T_BASE_SI,   "hbm_passive_ptrace.csv"))
    # 12 DRAM dies, each: Hybrid Bonding -> DRAM Die BEOL -> DRAM Die Si
    for t in range(args.tiers):
        t_si = T_DRAM_TOP if t == args.tiers - 1 else T_DRAM_SI
        layers.append(("hybrid_bond_flp.csv", T_HYBOND,    "hbm_passive_ptrace.csv"))
        layers.append(("dram_beol_flp.csv",   T_DRAM_BEOL, "hbm_passive_ptrace.csv"))
        layers.append(("dram_tier_flp.csv",   t_si,        "dram_tier_ptrace.csv"))
    # package cooling stack
    layers.append(("tim_flp.csv", T_TIM, "tim_ptrace.csv"))
    layers.append(("lid_flp.csv", T_LID, "lid_ptrace.csv"))

    lcf_path = "dram3d_3d_lcf.csv"
    write_lcf(layers, lcf_path)

    total_mem = 4 * args.hbm_per_stack
    print(f"\n  Layers: {len(layers)} device layers"
          f"  (5 GPU/interface + {0 if args.no_base_die else 2} base die"
          f" + {3*args.tiers} DRAM sublayers + TIM + Lid)"
          f"  + NoPackage HTC boundary on the Lid")
    print(f"  Filler: cross/frame = {filler}"
          f"  ({'k=140 high-k escape path (optimized)' if args.optimized else 'k=0.8 mold, no escape path (BASELINE thermal penalty)'})")
    print(f"  Power : GPU {args.gpu_power:g} W + HBM {total_mem:g} W"
          f" = {args.gpu_power + total_mem:g} W total")
    print(f"  Grid  : {args.grid}x{args.grid} on {args.gpu_side*1e3:g} mm die"
          f"  ({args.gpu_side/args.grid*1e3:.3f} mm/cell)")
    print(f"  LCF   : {lcf_path}\n")


if __name__ == "__main__":
    main()
