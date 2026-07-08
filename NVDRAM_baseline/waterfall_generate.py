"""
phi-HBM STCO Waterfall Generator  (NVDRAM_waterfall)
======================================================
Parameterized generator for the hybrid NVDRAM-bottom / DRAM-top stack, extended
so that each imec STCO intervention can be toggled independently. Built on the
EXP_NVDRAM_B base model (paper BASELINE = mold cross/frame, no thermal silicon);
the waterfall applies the interventions cumulatively.

Stack, bottom (adiabatic package side, layer 0) -> top (lid / cold-plate side):

    BSPDN -> GPU FEOL (heat source) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> [ HBM Base BEOL -> HBM Base Si ]                    (dropped by --no-base-die)
      -> nv_tiers   x [ Hybrid Bonding -> NVDRAM Die BEOL -> NVDRAM Die Si ]
      -> dram_tiers x [ Hybrid Bonding -> DRAM Die BEOL  -> DRAM Die Si  ]   (top die thick)
      -> TIM -> Lid
    (PACT then applies the NoPackage convective HTC boundary on top of the Lid)

Interventions (each maps to one STCO waterfall step):
  * --no-base-die   : HBM base-die removal   (drop the 2 HBM-base sublayers)
  * --merge-stacks  : adjacent-stack merging  (replace the inter-stack mold cross
                      with a continuous silicon region MERGE_SI, k=140, per the
                      imec paper -- "replacing the mold compound between them with
                      a continuous silicon region"; the four stacks stay memory,
                      only the cross material changes -> lateral Si bridge)
  * --thin-top-die  : top-die thinning        (top memory die Si 169 um -> 50 um)
  * --gpu-power W    : GPU frequency scaling   (linear P proportional to f; pass
                      0.7*414 = 289.8 for a 0.7x-frequency point)
  * --optimized      : thermal-silicon optimization (cross/frame filler MOLD -> THERMAL_SI)

Peak GPU temperature is reported by run_waterfall.py as the max under the 2x2
memory stacks (the framework's mask_surround convention: the mold gap cooks to a
few-hundred C as an artifact of the thin uniform-power GPU layer and is excluded).

Usage:
    python3 waterfall_generate.py [--nv-tiers N] [--dram-tiers N] [--gpu-power W]
        [--dram-per-stack W] [--dram-refresh-frac F] [--nv-leakage-factor F]
        [--gpu-side M] [--mem-side M] [--grid N]
        [--no-base-die] [--merge-stacks] [--thin-top-die] [--optimized]
"""

import argparse

# ---- defaults (paper-representative; identical to EXP_NVDRAM_B) ----
GPU_SIDE   = 0.024     # m  (24 mm GPU base die)
MEM_SIDE   = 0.010     # m  (10 mm memory-stack footprint)
GPU_POWER  = 414.0     # W  (GPU active heat source, 1.0f)
DRAM_PER_STACK = 40.0  # W  per 12-Hi stack (standby = leakage + refresh)
NV_TIERS   = 4         # NVDRAM dies per stack (bottom block)
DRAM_TIERS = 8         # DRAM dies per stack (top block)
GRID       = 48
EPS        = 1e-6

FILLER_BASELINE  = "MOLD"        # mold compound, k = 3 W/m-K (paper value)   (baseline)
FILLER_OPTIMIZED = "THERMAL_SI"  # high-k dummy silicon, k = 140              (optimized)
FILLER_MERGE     = "MERGE_SI"    # continuous silicon bridge, k = 140  (stack merging)

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
T_DRAM_TOP   = 169e-6     # top die Si (thinned to T_DIE_SI by --thin-top-die)
T_TIM        = 200e-6
T_LID        = 3000e-6


def gpu_block(gpu_side, label):
    return [(f"{label}", 0.0, 0.0, gpu_side, gpu_side, label)]


def mem_tier_blocks(gpu_side, mem_side, macro_label, filler_label, merge=False):
    """Memory-sublayer floorplan: 4 memory stacks (2x2) carrying `macro_label`,
    with a cross/frame of `filler_label` filling the rest.

    Stack merging (--merge-stacks) replaces the inter-stack mold *between* the
    stacks (the cross) with a continuous silicon region (`MERGE_SI`, k=140), per
    the imec paper. The four stacks stay memory; only the cross material changes,
    forming a continuous vertical/lateral silicon bridge between the stacks. The
    outer frame margin stays `filler_label` until the thermal-silicon step turns
    it (and everything) into THERMAL_SI. Returns (blocks, mem_names), no gaps.
    """
    g, h = gpu_side, mem_side
    gap = g - 2 * h
    margin = gap / 4.0
    central = gap / 2.0
    a0 = margin
    a1 = margin + h
    a2 = margin + h + central
    a3 = margin + h + central + h            # = g - margin
    cross_label = FILLER_MERGE if merge else filler_label   # continuous Si bridge when merged

    blocks = [
        ("MEM_BL", a0, a0, h, h, macro_label),
        ("MEM_BR", a2, a0, h, h, macro_label),
        ("MEM_TL", a0, a2, h, h, macro_label),
        ("MEM_TR", a2, a2, h, h, macro_label),
        ("Cross_V",  a1, margin, central, g - 2 * margin, cross_label),
        ("Cross_HL", a0, a1, h, central, cross_label),
        ("Cross_HR", a2, a1, h, central, cross_label),
        ("Frame_B", 0.0, 0.0, g, margin, filler_label),
        ("Frame_T", 0.0, a3, g, margin, filler_label),
        ("Frame_L", 0.0, margin, margin, g - 2 * margin, filler_label),
        ("Frame_R", a3, margin, margin, g - 2 * margin, filler_label),
    ]
    mem_names = ["MEM_BL", "MEM_BR", "MEM_TL", "MEM_TR"]
    return blocks, mem_names


def write_flp(path, blocks):
    lines = ["UnitName,X,Y,Length (m),Width (m),ConfigFile,Label"]
    for name, x, y, l, w, label in blocks:
        lines.append(f"{name},{x:g},{y:g},{l:g},{w:g},,{label}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_ptrace(path, names, power_map):
    lines = ["UnitName,Power,Power1,Power2"]
    for n in names:
        p = power_map.get(n, EPS)
        lines.append(f"{n},{p:g},{p:g},{p:g}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_lcf(layers, path):
    lines = ["Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow"]
    for idx, (flp, thickness, ptrace) in enumerate(layers):
        lines.append(f"{idx},{flp},{thickness:.3e},{ptrace},True")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Generate PACT inputs for one phi-HBM STCO waterfall stage.")
    p.add_argument("--nv-tiers", type=int, default=NV_TIERS)
    p.add_argument("--dram-tiers", type=int, default=DRAM_TIERS)
    p.add_argument("--gpu-power", type=float, default=GPU_POWER,
                   help="GPU heat-source power, W (frequency scaling: linear P proportional to f)")
    p.add_argument("--dram-per-stack", type=float, default=DRAM_PER_STACK)
    p.add_argument("--dram-refresh-frac", type=float, default=DRAM_REFRESH_FRAC)
    p.add_argument("--nv-leakage-factor", type=float, default=NV_LEAKAGE_FACTOR)
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE)
    p.add_argument("--mem-side", type=float, default=MEM_SIDE)
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--no-base-die", action="store_true",
                   help="HBM base-die removal: drop the 2 HBM-base sublayers")
    p.add_argument("--merge-stacks", action="store_true",
                   help="Adjacent-stack merging: 2x2 macros + cross -> one contiguous memory slab")
    p.add_argument("--thin-top-die", action="store_true",
                   help="Top-die thinning: top memory die Si 169 um -> 50 um")
    p.add_argument("--optimized", action="store_true",
                   help="Thermal-silicon optimization: cross/frame filler MOLD -> THERMAL_SI")
    args = p.parse_args()

    if args.mem_side * 2 >= args.gpu_side:
        raise SystemExit("Need 2*mem-side < gpu-side so the 2x2 memory array fits with margins.")

    filler = FILLER_OPTIMIZED if args.optimized else FILLER_BASELINE
    top_si = T_DIE_SI if args.thin_top_die else T_DRAM_TOP
    merge = args.merge_stacks

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

    # ---- memory-stack sublayer floorplans (2x2 macros or merged slab + filler) ----
    base_beol, _   = mem_tier_blocks(args.gpu_side, args.mem_side, "HBM_BASE_BEOL", filler, merge)
    base_si, _     = mem_tier_blocks(args.gpu_side, args.mem_side, "HBM_BASE_SI",   filler, merge)
    hybond, _      = mem_tier_blocks(args.gpu_side, args.mem_side, "HYBRID_BOND",   filler, merge)
    nv_beol, _     = mem_tier_blocks(args.gpu_side, args.mem_side, "NV_DIE_BEOL",   filler, merge)
    nv_si, mem_nm  = mem_tier_blocks(args.gpu_side, args.mem_side, "NV_DIE_SI",     filler, merge)
    dram_beol, _   = mem_tier_blocks(args.gpu_side, args.mem_side, "DRAM_BEOL",     filler, merge)
    dram_si, _     = mem_tier_blocks(args.gpu_side, args.mem_side, "DRAM_SI",       filler, merge)
    all_names = [b[0] for b in dram_si]
    write_flp("hbm_base_beol_flp.csv", base_beol)
    write_flp("hbm_base_si_flp.csv",   base_si)
    write_flp("hybrid_bond_flp.csv",   hybond)
    write_flp("nv_die_beol_flp.csv",   nv_beol)
    write_flp("nv_tier_flp.csv",       nv_si)
    write_flp("dram_die_beol_flp.csv", dram_beol)
    write_flp("dram_tier_flp.csv",     dram_si)

    # Standby-power decomposition (background only), REPLACEMENT semantics.
    # Each of the 4 macros per active tier carries per_die; merging changes only
    # the cross material, not the macro count, so total memory power is preserved.
    total_tiers      = args.nv_tiers + args.dram_tiers
    dram_per_die     = args.dram_per_stack / total_tiers
    dram_leakage_die = dram_per_die * (1.0 - args.dram_refresh_frac)
    nv_per_die       = args.nv_leakage_factor * dram_leakage_die
    write_ptrace("nv_tier_ptrace.csv",   all_names, {n: nv_per_die   for n in mem_nm})
    write_ptrace("dram_tier_ptrace.csv", all_names, {n: dram_per_die for n in mem_nm})
    write_ptrace("mem_passive_ptrace.csv", all_names, {})

    # ---- assemble the layer stack (bottom -> top) ----
    layers = [
        ("bspdn_flp.csv",    T_BSPDN,    "bspdn_ptrace.csv"),
        ("gpu_feol_flp.csv", T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("beol_mxy_flp.csv", T_BEOL_MXY, "beol_mxy_ptrace.csv"),
        ("oxide_flp.csv",    T_OXIDE,    "oxide_ptrace.csv"),
        ("ubump_flp.csv",    T_UBUMP,    "ubump_ptrace.csv"),
    ]
    if not args.no_base_die:
        layers.append(("hbm_base_beol_flp.csv", T_BASE_BEOL, "mem_passive_ptrace.csv"))
        layers.append(("hbm_base_si_flp.csv",   T_BASE_SI,   "mem_passive_ptrace.csv"))
    # NVDRAM tiers (bottom)
    for i in range(args.nv_tiers):
        nv_is_top = (args.dram_tiers == 0 and i == args.nv_tiers - 1)
        t_si = top_si if nv_is_top else T_DIE_SI
        layers.append(("hybrid_bond_flp.csv", T_HYBOND,   "mem_passive_ptrace.csv"))
        layers.append(("nv_die_beol_flp.csv", T_DIE_BEOL, "mem_passive_ptrace.csv"))
        layers.append(("nv_tier_flp.csv",     t_si,       "nv_tier_ptrace.csv"))
    # DRAM tiers (top); last die = top die
    for t in range(args.dram_tiers):
        t_si = top_si if t == args.dram_tiers - 1 else T_DIE_SI
        layers.append(("hybrid_bond_flp.csv",   T_HYBOND,   "mem_passive_ptrace.csv"))
        layers.append(("dram_die_beol_flp.csv", T_DIE_BEOL, "mem_passive_ptrace.csv"))
        layers.append(("dram_tier_flp.csv",     t_si,       "dram_tier_ptrace.csv"))
    layers.append(("tim_flp.csv", T_TIM, "tim_ptrace.csv"))
    layers.append(("lid_flp.csv", T_LID, "lid_ptrace.csv"))

    write_lcf(layers, "hybrid_lcf.csv")

    nv_total   = 4 * nv_per_die * args.nv_tiers
    dram_total = 4 * dram_per_die * args.dram_tiers
    ints = [s for s, on in [("base-die-removal", args.no_base_die),
                            ("stack-merging", merge),
                            ("top-die-thinning", args.thin_top_die),
                            ("thermal-silicon", args.optimized)] if on]
    print(f"[waterfall_generate] {len(layers)} layers | filler={filler} | "
          f"gpu={args.gpu_power:g} W | top_si={top_si*1e6:g} um | "
          f"interventions={ints if ints else ['none (baseline)']} | "
          f"mem power {nv_total+dram_total:.1f} W")


if __name__ == "__main__":
    main()
