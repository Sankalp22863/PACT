"""
Hybrid NVDRAM-bottom / DRAM-top 3D-Integration Generator (paper BASELINE, no thermal silicon)
=============================================================================================
Same detailed hybrid stack as EXP_NVDRAM_S, but this variant reproduces the
paper's *un-optimized* baseline (the "3D thermal penalty" regime of the IEDM
STCO staircase, "Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via
System-Technology Co-Optimization", imec, IEDM 2025).

The ONLY difference from EXP_NVDRAM_S is the cross/frame filler material between
and around the four memory stacks:

  * EXP_NVDRAM_S (optimized) : cross/frame = THERMAL_SI  (k = 140 W/m-K)
                               -> high-k vertical/lateral heat-escape paths
                               -> "thermal silicon optimization" already applied.
  * EXP_NVDRAM_B (baseline)  : cross/frame = MOLD         (k = 0.8 W/m-K)
                               -> standard epoxy mold compound; no escape path,
                                  heat forced up through the low-k memory-die
                                  BEOL / hybrid-bond bottleneck (thermal penalty).

Stack, bottom (adiabatic package side) -> top (lid):

    BSPDN -> GPU FEOL (heat source, 414 W) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> HBM Base BEOL -> HBM Base Si (Base Die)
      -> nv_tiers   x [ Hybrid Bonding -> NVDRAM Die BEOL -> NVDRAM Die Si ]
      -> dram_tiers x [ Hybrid Bonding -> DRAM Die BEOL  -> DRAM Die Si  ]  (top die 169 um)
      -> TIM -> Lid
    (PACT applies the NoPackage convective HTC boundary on top of the Lid)

Memory power = standby only (REPLACEMENT semantics): the memory is a fixed
(nv_tiers + dram_tiers)-Hi stack carrying the paper's dram_per_stack budget, so
each DRAM die = dram_per_stack / total_tiers. NVDRAM dies replace that many DRAM
dies, each dissipating  nv_leakage_factor * (DRAM leakage)  + 0 refresh.

Usage:
    python generate.py [--nv-tiers N] [--dram-tiers N] [--gpu-power W]
                       [--dram-per-stack W] [--dram-refresh-frac F]
                       [--nv-leakage-factor F] [--gpu-side M] [--mem-side M]
                       [--optimized]     # use THERMAL_SI instead of MOLD (= EXP_NVDRAM_S)
"""

import argparse

# ---- defaults ----
GPU_SIDE   = 0.024     # m  (24 mm GPU base die)
MEM_SIDE   = 0.010     # m  (10 mm memory-stack footprint)
GPU_POWER  = 414.0     # W  (GPU active heat source)
DRAM_PER_STACK = 40.0  # W  per 12-Hi stack (standby = leakage + refresh)
NV_TIERS   = 4         # NVDRAM dies per stack (bottom block)
DRAM_TIERS = 8         # DRAM dies per stack (top block)
GRID       = 48
EPS        = 1e-6

# ---- cross/frame filler: BASELINE uses mold (no thermal silicon) ----
FILLER_BASELINE  = "MOLD"        # epoxy mold compound, k ~ 0.8 W/m-K  (paper baseline)
FILLER_OPTIMIZED = "THERMAL_SI"  # high-k dummy silicon, k = 140       (= EXP_NVDRAM_S)

# ---- memory standby-power model (background only: leakage + refresh) ----
DRAM_REFRESH_FRAC = 0.35   # fraction of DRAM standby power spent on refresh
NV_LEAKAGE_FACTOR = 0.5    # NVDRAM leakage as a fraction of DRAM leakage (refresh = 0)

# ---- layer thicknesses (m), from the paper cross-section table ----
T_BSPDN      = 1.715e-6
T_GPU_FEOL   = 0.15e-6
T_BEOL_MXY   = 1.4e-6
T_OXIDE      = 1.0e-6
T_UBUMP      = 40e-6
T_BASE_BEOL  = 5e-6
T_BASE_SI    = 50e-6
T_HYBOND     = 2e-6
T_DIE_BEOL   = 3e-6
T_DIE_SI     = 50e-6      # inner die Si (thinned)
T_DRAM_TOP   = 169e-6     # top DRAM die Si
T_TIM        = 200e-6
T_LID        = 3000e-6


def gpu_block(gpu_side, label):
    return [(f"{label}", 0.0, 0.0, gpu_side, gpu_side, label)]


def mem_tier_blocks(gpu_side, mem_side, macro_label, filler_label):
    """4 memory stacks in 2x2 + cross/frame filler filling the rest.

    The four stack footprints carry `macro_label`; the cross/frame is
    `filler_label` -- THERMAL_SI (k=140, optimized) or MOLD (k=0.8, baseline).
    """
    g, h = gpu_side, mem_side
    gap = g - 2 * h
    margin = gap / 4.0
    central = gap / 2.0
    a0 = margin
    a1 = margin + h
    a2 = margin + h + central
    a3 = margin + h + central + h

    blocks = []
    blocks.append(("MEM_BL", a0, a0, h, h, macro_label))
    blocks.append(("MEM_BR", a2, a0, h, h, macro_label))
    blocks.append(("MEM_TL", a0, a2, h, h, macro_label))
    blocks.append(("MEM_TR", a2, a2, h, h, macro_label))
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
    p = argparse.ArgumentParser(
        description="Generate PACT inputs for the BASELINE hybrid NVDRAM-bottom / DRAM-top stack (no thermal silicon).")
    p.add_argument("--nv-tiers", type=int, default=NV_TIERS)
    p.add_argument("--dram-tiers", type=int, default=DRAM_TIERS)
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--dram-per-stack", type=float, default=DRAM_PER_STACK,
                   help="DRAM standby power per stack, W (leakage + refresh)")
    p.add_argument("--dram-refresh-frac", type=float, default=DRAM_REFRESH_FRAC,
                   help="fraction of DRAM standby power spent on refresh (NVDRAM has none)")
    p.add_argument("--nv-leakage-factor", type=float, default=NV_LEAKAGE_FACTOR,
                   help="NVDRAM leakage as a fraction of DRAM leakage (refresh = 0)")
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE)
    p.add_argument("--mem-side", type=float, default=MEM_SIDE)
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--optimized", action="store_true",
                   help="Use THERMAL_SI (k=140) cross/frame instead of MOLD -> reproduces EXP_NVDRAM_S")
    args = p.parse_args()

    if args.mem_side * 2 >= args.gpu_side:
        raise SystemExit("Need 2*mem-side < gpu-side so the 2x2 memory array fits with margins.")

    filler = FILLER_OPTIMIZED if args.optimized else FILLER_BASELINE

    print(f"\nGenerating {'OPTIMIZED' if args.optimized else 'BASELINE'} hybrid NVDRAM-bottom / DRAM-top stack: "
          f"GPU {args.gpu_side*1e3:g} mm @ {args.gpu_power:g} W, "
          f"{args.nv_tiers} NVDRAM + {args.dram_tiers} DRAM tiers (4 stacks, 2x2);"
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

    # ---- memory-stack sublayer floorplans (2x2 macros + cross/frame filler) ----
    base_beol = mem_tier_blocks(args.gpu_side, args.mem_side, "HBM_BASE_BEOL", filler)
    base_si   = mem_tier_blocks(args.gpu_side, args.mem_side, "HBM_BASE_SI", filler)
    hybond    = mem_tier_blocks(args.gpu_side, args.mem_side, "HYBRID_BOND", filler)
    nv_beol   = mem_tier_blocks(args.gpu_side, args.mem_side, "NV_DIE_BEOL", filler)
    nv_si     = mem_tier_blocks(args.gpu_side, args.mem_side, "NV_DIE_SI", filler)
    dram_beol = mem_tier_blocks(args.gpu_side, args.mem_side, "DRAM_BEOL", filler)
    dram_si   = mem_tier_blocks(args.gpu_side, args.mem_side, "DRAM_SI", filler)
    names = [b[0] for b in dram_si]
    write_flp("hbm_base_beol_flp.csv", base_beol)
    write_flp("hbm_base_si_flp.csv",   base_si)
    write_flp("hybrid_bond_flp.csv",   hybond)
    write_flp("nv_die_beol_flp.csv",   nv_beol)
    write_flp("nv_tier_flp.csv",       nv_si)     # NVDRAM Die Si (the NV "tier")
    write_flp("dram_die_beol_flp.csv", dram_beol)
    write_flp("dram_tier_flp.csv",     dram_si)   # DRAM Die Si (the DRAM "tier")

    # Standby-power decomposition (background only), REPLACEMENT semantics:
    #   DRAM die  = leakage + refresh                       (= dram_per_stack/total_tiers)
    #   NVDRAM die= nv_leakage_factor * leakage  +  0 refresh
    total_tiers      = args.nv_tiers + args.dram_tiers
    dram_per_die     = args.dram_per_stack / total_tiers
    dram_leakage_die = dram_per_die * (1.0 - args.dram_refresh_frac)
    dram_refresh_die = dram_per_die * args.dram_refresh_frac
    nv_per_die       = args.nv_leakage_factor * dram_leakage_die
    write_ptrace("nv_tier_ptrace.csv",   names, {n: nv_per_die for n in names if n.startswith("MEM_")})
    write_ptrace("dram_tier_ptrace.csv", names, {n: dram_per_die for n in names if n.startswith("MEM_")})
    write_ptrace("mem_passive_ptrace.csv", names, {})

    # ---- assemble the layer stack (bottom -> top) ----
    layers = [
        ("bspdn_flp.csv",    T_BSPDN,    "bspdn_ptrace.csv"),
        ("gpu_feol_flp.csv", T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("beol_mxy_flp.csv", T_BEOL_MXY, "beol_mxy_ptrace.csv"),
        ("oxide_flp.csv",    T_OXIDE,    "oxide_ptrace.csv"),
        ("ubump_flp.csv",    T_UBUMP,    "ubump_ptrace.csv"),
        ("hbm_base_beol_flp.csv", T_BASE_BEOL, "mem_passive_ptrace.csv"),
        ("hbm_base_si_flp.csv",   T_BASE_SI,   "mem_passive_ptrace.csv"),
    ]
    # NVDRAM tiers (bottom): Hybrid Bonding -> NVDRAM Die BEOL -> NVDRAM Die Si.
    # When there are no DRAM tiers on top, the top-most NVDRAM die is the thick
    # top die (T_DRAM_TOP) so the total stack height matches the DRAM cases.
    for i in range(args.nv_tiers):
        nv_is_top = (args.dram_tiers == 0 and i == args.nv_tiers - 1)
        t_si = T_DRAM_TOP if nv_is_top else T_DIE_SI
        layers.append(("hybrid_bond_flp.csv", T_HYBOND,  "mem_passive_ptrace.csv"))
        layers.append(("nv_die_beol_flp.csv", T_DIE_BEOL,"mem_passive_ptrace.csv"))
        layers.append(("nv_tier_flp.csv",     t_si,      "nv_tier_ptrace.csv"))
    # DRAM tiers (top): Hybrid Bonding -> DRAM Die BEOL -> DRAM Die Si (last = top die)
    for t in range(args.dram_tiers):
        t_si = T_DRAM_TOP if t == args.dram_tiers - 1 else T_DIE_SI
        layers.append(("hybrid_bond_flp.csv",   T_HYBOND,   "mem_passive_ptrace.csv"))
        layers.append(("dram_die_beol_flp.csv", T_DIE_BEOL, "mem_passive_ptrace.csv"))
        layers.append(("dram_tier_flp.csv",     t_si,       "dram_tier_ptrace.csv"))
    layers.append(("tim_flp.csv", T_TIM, "tim_ptrace.csv"))
    layers.append(("lid_flp.csv", T_LID, "lid_ptrace.csv"))

    write_lcf(layers, "hybrid_lcf.csv")

    nv_total   = 4 * nv_per_die * args.nv_tiers
    dram_total = 4 * dram_per_die * args.dram_tiers
    all_dram   = 4 * dram_per_die * total_tiers
    print(f"\n  Layers: {len(layers)} device layers"
          f"  (5 GPU/interface + 2 base die + {3*args.nv_tiers} NVDRAM + {3*args.dram_tiers} DRAM sublayers + TIM + Lid)")
    print(f"  Filler: cross/frame = {filler}"
          f"  ({'k=140 high-k escape path (optimized)' if args.optimized else 'k=0.8 mold, no escape path (BASELINE thermal penalty)'})")
    print(f"  Memory standby power (per die):")
    print(f"    DRAM   = {dram_per_die:.2f} W  (leakage {dram_leakage_die:.2f} + refresh {dram_refresh_die:.2f})")
    print(f"    NVDRAM = {nv_per_die:.2f} W  ({args.nv_leakage_factor:g}x DRAM leakage + 0 refresh)  -> < DRAM")
    print(f"  Power : GPU {args.gpu_power:g} W + memory {nv_total+dram_total:g} W"
          f" (NVDRAM {nv_total:g} + DRAM {dram_total:g}) = {args.gpu_power+nv_total+dram_total:g} W total")
    print(f"  Replacement check: hybrid memory {nv_total+dram_total:.1f} W < all-DRAM {all_dram:.1f} W"
          f"  (bottom {args.nv_tiers} tiers swapped DRAM->NVDRAM)")
    print(f"  LCF   : hybrid_lcf.csv\n")


if __name__ == "__main__":
    main()
