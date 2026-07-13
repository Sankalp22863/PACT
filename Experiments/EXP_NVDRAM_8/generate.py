"""
Hybrid NVDRAM-bottom / DRAM-top Generator (paper-baseline conditions: unmerged stacks + cluster map)
====================================================================================================
Same detailed hybrid stack as EXP_NVDRAM_S, but this variant reproduces the
paper's *un-optimized* baseline (the "3D thermal penalty" regime of the IEDM
STCO staircase, "Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via
System-Technology Co-Optimization", imec, IEDM 2025).

Like EXP_DRAM_B, this variant models the paper's baseline two ways in which it
differs from the merged/optimized case (EXP_NVDRAM_S):

  1. UNMERGED STACKS: a thin mold gap separates the two memory stacks on each
     short edge (paper: ~0.1 mm of k=3 mold; modelled 1.0 mm wide with
     k_eff = 30 W/m-K to preserve the lateral resistance -- see [MOLD]).
  2. NON-UNIFORM GPU POWER: a structured cluster power map (see
     scripts/stack_common.py -- correlated seeded-random cluster utilisation,
     asymmetric activity tilt, powered central L2/NoC column, IO edge ring);
     tile peak-to-average set by --map-p2a, default calibrated on EXP_DRAM_B
     to the paper's 141.7 C baseline.

The central column stays THERMAL_SI (part of the paper's baseline already).
All layers are embedded in a molded package surround (PKG_MOLD ring + extended
copper lid) so heat can spread laterally beyond the die.
--merged and --power-map uniform together reproduce EXP_NVDRAM_S.

Stack, bottom (adiabatic package side) -> top (lid):

    BSPDN -> GPU FEOL (heat source, 414 W) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> HBM Base BEOL -> HBM Base Si (Base Die)
      -> nv_tiers   x [ Hybrid Bonding -> NVDRAM Die BEOL -> NVDRAM Die Si ]
      -> dram_tiers x [ Hybrid Bonding -> DRAM Die BEOL  -> DRAM Die Si  ]  (top die 169 um)
      -> TIM -> Lid
    (PACT applies the NoPackage convective HTC boundary on top of the Lid)

Memory power (REPLACEMENT semantics): the memory is a fixed
(nv_tiers + dram_tiers)-Hi stack. Each DRAM die carries the paper's workload
budget share (dram_per_stack / total_tiers; DRAM access energy is implicit in
it). NVDRAM dies replace that many DRAM dies, each dissipating
  nv_leakage_factor * (DRAM leakage)  +  0 refresh
  +  explicit core-array access power = nv_access_fj x (stack_bw / total_tiers)
(device numbers: read == write ~90 fJ/bit, destructive read; range 50-200).

Usage:
    python generate.py [--nv-tiers N] [--dram-tiers N] [--gpu-power W]
                       [--dram-per-stack W] [--dram-refresh-frac F]
                       [--nv-leakage-factor F] [--gpu-len M] [--gpu-wid M]
                       [--mem-side M] [--gap-wid M] [--merged] [--pkg-margin M]
                       [--power-map {cluster,uniform}] [--map-p2a R] [--map-seed N]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import stack_common as sc  # noqa: E402

# ---- defaults (paper Fig. 3 die layout) ----
GPU_LEN    = 0.030     # m  (30 mm GPU die, long side = x; 11 + 8 + 11)
GPU_WID    = 0.022     # m  (22 mm GPU die, short side = y; 11 + 11)
MEM_SIDE   = 0.011     # m  (11 mm memory-stack footprint)
GAP_WID    = 0.001     # m  (grid-resolvable stand-in for the paper's 0.1 mm mold
                       #     gap between adjacent stacks; k scaled to compensate)

MAP_P2A    = 2.19      # cluster-map tile peak-to-average ratio (calibrated on
                       #     EXP_DRAM_B against the paper's 141.7 C GPU peak)
GPU_POWER  = 414.0     # W  (GPU active heat source)
DRAM_PER_STACK = 40.0  # W  per 12-Hi stack (standby = leakage + refresh)
NV_TIERS   = 4         # NVDRAM dies per stack (bottom block)
DRAM_TIERS = 8         # DRAM dies per stack (top block)
GRID       = 48
EPS        = 1e-6

# ---- fillers: central column is thermal Si (paper baseline); the baseline
# penalty lives in the inter-stack MOLD gap (equivalent-k, see config) ----
FILLER_CENTER = "THERMAL_SI"
FILLER_GAP    = "MOLD"

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


def gpu_block(gpu_len, gpu_wid, label):
    return [(f"{label}", 0.0, 0.0, gpu_len, gpu_wid, label)]


def mem_tier_blocks(gpu_len, gpu_wid, mem_side, gap_wid, macro_label):
    """Paper Fig. 3(a) baseline layout: 4 memory stacks in the die corners (two
    per short edge), a mold gap of `gap_wid` between the two stacks on each edge
    (the un-merged baseline; 0 = merged), and the central THERMAL_SI column."""
    h = mem_side
    central = gpu_len - 2 * h            # central column width (paper: 8 mm)
    sy = (gpu_wid - gap_wid) / 2.0       # stack extent in y
    xr = gpu_len - h                     # right stack-column X
    blocks = [
        ("MEM_BL", 0.0, 0.0,          h, sy, macro_label),
        ("MEM_TL", 0.0, sy + gap_wid, h, sy, macro_label),
        ("MEM_BR", xr,  0.0,          h, sy, macro_label),
        ("MEM_TR", xr,  sy + gap_wid, h, sy, macro_label),
        ("Center", h, 0.0, central, gpu_wid, FILLER_CENTER),
    ]
    if gap_wid > 1e-9:
        blocks += [
            ("Gap_L", 0.0, sy, h, gap_wid, FILLER_GAP),
            ("Gap_R", xr,  sy, h, gap_wid, FILLER_GAP),
        ]
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
    p.add_argument("--nv-access-fj", type=float, default=sc.NV_ACCESS_FJ,
                   help="NVDRAM core-array access energy, fJ/bit (read == write; device range 50-200)")
    p.add_argument("--stack-bw", type=float, default=sc.STACK_BW_TBS,
                   help="memory bandwidth per stack, TB/s (default = paper's 4x HBM3E from 3D)")
    p.add_argument("--bw-util", type=float, default=sc.BW_UTIL,
                   help="sustained fraction of peak bandwidth")
    p.add_argument("--gpu-len", type=float, default=GPU_LEN,
                   help="GPU die length (x, along the stack columns), m")
    p.add_argument("--gpu-wid", type=float, default=GPU_WID,
                   help="GPU die width (y, the short edges carrying the stacks), m")
    p.add_argument("--mem-side", type=float, default=MEM_SIDE)
    p.add_argument("--gap-wid", type=float, default=GAP_WID,
                   help="mold gap between the two stacks on each edge (m); pair with the equivalent-k [MOLD] entry")
    p.add_argument("--merged", action="store_true",
                   help="Remove the inter-stack mold gaps (the paper's 'HBM stack merging')")
    p.add_argument("--pkg-margin", type=float, default=sc.PKG_MARGIN,
                   help="package mold ring width around the die (m); grid in modelParams must cover die+2*margin")
    p.add_argument("--power-map", choices=["cluster", "uniform"], default="cluster",
                   help="GPU FEOL power: structured cluster map (paper-like) or uniform")
    p.add_argument("--map-p2a", type=float, default=MAP_P2A,
                   help="tile peak-to-average power-density ratio of the cluster map")
    p.add_argument("--map-seed", type=int, default=sc.MAP_SEED,
                   help="seed for the cluster map's utilisation pattern")
    p.add_argument("--grid", type=int, default=GRID)
    args = p.parse_args()

    if args.mem_side * 2 >= args.gpu_len:
        raise SystemExit("Need 2*mem-side < gpu-len so a central column separates the stacks.")
    if args.mem_side * 2 > args.gpu_wid + 1e-12:
        raise SystemExit("Need 2*mem-side <= gpu-wid so two stacks fit along each short edge.")
    gap = 0.0 if args.merged else args.gap_wid
    m = args.pkg_margin

    def ringed(blocks):
        return sc.add_package_ring(blocks, args.gpu_len, args.gpu_wid, m)

    def passive_layer(flp_path, ptrace_path, blocks):
        write_flp(flp_path, blocks)
        write_ptrace(ptrace_path, [b[0] for b in blocks], {})

    print(f"\nGenerating BASELINE hybrid NVDRAM-bottom / DRAM-top stack: "
          f"GPU {args.gpu_len*1e3:g} x {args.gpu_wid*1e3:g} mm @ {args.gpu_power:g} W "
          f"({args.power_map} power map), package "
          f"{(args.gpu_len+2*m)*1e3:g} x {(args.gpu_wid+2*m)*1e3:g} mm, "
          f"{args.nv_tiers} NVDRAM + {args.dram_tiers} DRAM tiers (4 corner stacks);"
          f" inter-stack gap = {gap*1e3:g} mm {FILLER_GAP if gap else '(merged)'}.\n")

    # ---- GPU + interface + package floorplans (die + PKG_MOLD ring) ----
    passive_layer("bspdn_flp.csv",    "bspdn_ptrace.csv",
                  ringed(gpu_block(args.gpu_len, args.gpu_wid, "BSPDN")))
    passive_layer("beol_mxy_flp.csv", "beol_mxy_ptrace.csv",
                  ringed(gpu_block(args.gpu_len, args.gpu_wid, "BEOL_MXY")))
    passive_layer("oxide_flp.csv",    "oxide_ptrace.csv",
                  ringed(gpu_block(args.gpu_len, args.gpu_wid, "OXIDE")))
    passive_layer("ubump_flp.csv",    "ubump_ptrace.csv",
                  ringed(gpu_block(args.gpu_len, args.gpu_wid, "GPU_HBM_UBUMP")))
    passive_layer("tim_flp.csv",      "tim_ptrace.csv",
                  ringed(gpu_block(args.gpu_len, args.gpu_wid, "TIM")))
    # the copper cold-plate lid spans the whole package
    passive_layer("lid_flp.csv", "lid_ptrace.csv",
                  sc.full_package_block(args.gpu_len, args.gpu_wid, m, "LID"))

    # GPU FEOL: structured cluster map, or a uniform single die block
    if args.power_map == "uniform":
        feol = ringed(gpu_block(args.gpu_len, args.gpu_wid, "GPU_FEOL"))
        write_flp("gpu_feol_flp.csv", feol)
        write_ptrace("gpu_feol_ptrace.csv", [b[0] for b in feol],
                     {"GPU_FEOL": args.gpu_power})
    else:
        tiles, tile_power = sc.cluster_power_tiles(args.gpu_len, args.gpu_wid,
                                                   args.gpu_power, args.map_p2a,
                                                   seed=args.map_seed)
        feol = ringed(tiles)
        write_flp("gpu_feol_flp.csv", feol)
        write_ptrace("gpu_feol_ptrace.csv", [b[0] for b in feol], tile_power)
        avg = args.gpu_power / len(tiles)
        print(f"  Cluster map: {len(tiles)} tiles of {sc.POWER_TILE*1e3:g} mm, "
              f"peak {max(tile_power.values()):.3f} W / avg {avg:.3f} W per tile "
              f"(p2a = {max(tile_power.values())/avg:.2f}, seed {args.map_seed})")

    # ---- memory-stack sublayers (corner stacks + gaps + central Si + ring) ----
    base_beol = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "HBM_BASE_BEOL"))
    base_si   = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "HBM_BASE_SI"))
    hybond    = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "HYBRID_BOND"))
    nv_beol   = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "NV_DIE_BEOL"))
    nv_si     = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "NV_DIE_SI"))
    dram_beol = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "DRAM_BEOL"))
    dram_si   = ringed(mem_tier_blocks(args.gpu_len, args.gpu_wid, args.mem_side, gap, "DRAM_SI"))
    names = [b[0] for b in dram_si]
    write_flp("hbm_base_beol_flp.csv", base_beol)
    write_flp("hbm_base_si_flp.csv",   base_si)
    write_flp("hybrid_bond_flp.csv",   hybond)
    write_flp("nv_die_beol_flp.csv",   nv_beol)
    write_flp("nv_tier_flp.csv",       nv_si)     # NVDRAM Die Si (the NV "tier")
    write_flp("dram_die_beol_flp.csv", dram_beol)
    write_flp("dram_tier_flp.csv",     dram_si)   # DRAM Die Si (the DRAM "tier")

    # Power decomposition, REPLACEMENT semantics:
    #   DRAM die  = leakage + refresh  (= dram_per_stack/total_tiers; the paper's
    #               workload budget, so DRAM access energy is implicit in it)
    #   NVDRAM die= nv_leakage_factor * (DRAM leakage) + 0 refresh
    #               + explicit core-array access power (device numbers:
    #                 fJ/bit x per-die bandwidth; read == write, destructive read)
    total_tiers      = args.nv_tiers + args.dram_tiers
    dram_per_die     = args.dram_per_stack / total_tiers
    dram_leakage_die = dram_per_die * (1.0 - args.dram_refresh_frac)
    dram_refresh_die = dram_per_die * args.dram_refresh_frac
    nv_access_die    = sc.nv_access_power_per_die(args.nv_access_fj, args.stack_bw,
                                                  args.bw_util, total_tiers)
    nv_per_die       = args.nv_leakage_factor * dram_leakage_die + nv_access_die
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
    print(f"  Layout: central {FILLER_CENTER} column; inter-stack gap"
          f" {gap*1e3:g} mm {FILLER_GAP} (equivalent-k for the paper's 0.1 mm)"
          if gap else "  Layout: central THERMAL_SI column; stacks merged")
    print(f"  Ring  : {m*1e3:g} mm {sc.RING_LABEL} package surround, lid spans the package")
    print(f"  GPU map: {args.power_map}"
          + (f" (p2a = {args.map_p2a:g}, seed {args.map_seed})" if args.power_map == "cluster" else ""))
    print(f"  Memory power (per die):")
    print(f"    DRAM   = {dram_per_die:.2f} W  (leakage {dram_leakage_die:.2f} + refresh {dram_refresh_die:.2f};"
          f" access implicit in the workload budget)")
    print(f"    NVDRAM = {nv_per_die:.2f} W  ({args.nv_leakage_factor:g}x DRAM leakage = "
          f"{args.nv_leakage_factor*dram_leakage_die:.2f} + access {nv_access_die:.2f}"
          f" @ {args.nv_access_fj:g} fJ/bit x {args.stack_bw:g} TB/s x {args.bw_util:g} util / {total_tiers} dies;"
          f" 0 refresh)")
    print(f"  Power : GPU {args.gpu_power:g} W + memory {nv_total+dram_total:g} W"
          f" (NVDRAM {nv_total:g} + DRAM {dram_total:g}) = {args.gpu_power+nv_total+dram_total:g} W total")
    print(f"  Replacement check: hybrid memory {nv_total+dram_total:.1f} W < all-DRAM {all_dram:.1f} W"
          f"  (bottom {args.nv_tiers} tiers swapped DRAM->NVDRAM)")
    print(f"  LCF   : hybrid_lcf.csv\n")


if __name__ == "__main__":
    main()
