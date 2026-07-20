"""
3D HBM-on-GPU Stack Generator  (paper BASELINE = "3D thermal penalty", 141.7 C)
===============================================================================
Same detailed HBM-on-GPU stack as EXP_DRAM_R, but this variant reproduces the
paper's *un-optimized* 3D baseline -- the leftmost "3D thermal penalty" point of
the IEDM STCO staircase (GPU peak 141.7 C in "Breaking Thermal Bottleneck in 3D
HBM-on-GPU Integration via System-Technology Co-Optimization", imec, IEDM 2025).

The paper's baseline differs from the merged/optimized case (EXP_DRAM_R) in two
ways, both modelled here:

  1. UNMERGED STACKS: the two adjacent HBM stacks on each short edge are
     separated by a thin mold gap (~0.1 mm of k=3 W/m-K mold in the paper).
     The 0.5 mm grid cannot resolve 0.1 mm, so the gap is modelled 1.0 mm wide
     with k scaled x10 (k_eff = 30 W/m-K) to preserve the same lateral thermal
     resistance (see [MOLD] in experiment.config).
  2. NON-UNIFORM GPU POWER: the paper applies commercial 0.5 mm-resolution power
     maps; those are not public, so a structured stand-in is used (see
     scripts/stack_common.py): a compute-cluster lattice with correlated
     seeded-random utilisation and an asymmetric activity tilt, a powered
     central L2/NoC column, and a low-power IO ring, at 0.5 mm tiles. The tile
     peak-to-average ratio (--map-p2a) is the single knob, calibrated so this
     baseline reproduces the paper's 141.7 C GPU peak.

The central column between the stack pairs is THERMAL_SI in the paper's baseline
already ("thermal silicon fills the central void"), so it stays THERMAL_SI here.
All layers are embedded in a molded package surround (PKG_MOLD ring + extended
copper lid, see stack_common.add_package_ring) so heat can spread laterally
beyond the die as in the paper's 65x65 mm package.
--merged removes the mold gaps and --power-map uniform restores the flat 414 W
map (together those reproduce EXP_DRAM_R).

Stack, bottom (adiabatic package side, layer 0) -> top (lid / cold-plate side):

    BSPDN -> GPU FEOL (heat source, 414 W) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> HBM Base BEOL -> HBM Base Si (Base Die)
      -> 12 x [ Hybrid Bonding -> DRAM Die BEOL -> DRAM Die Si ]   (top die = 169 um)
      -> TIM -> Lid
    (PACT then applies the NoPackage convective HTC boundary on top of the Lid)

Usage:
    python generate.py [--gpu-power W] [--hbm-per-stack W] [--tiers N]
                       [--gpu-len M] [--gpu-wid M] [--hbm-side M] [--no-base-die]
                       [--gap-wid M] [--merged] [--pkg-margin M]
                       [--power-map {cluster,uniform}] [--map-p2a R] [--map-seed N]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import stack_common as sc  # noqa: E402

# ---- defaults (paper Fig. 3 die layout) ----
GPU_LEN    = 0.030     # m  (30 mm GPU die, long side = x; 11 + 8 + 11)
GPU_WID    = 0.022     # m  (22 mm GPU die, short side = y)
HBM_SIDE   = 0.011     # m  (11 mm HBM stack footprint, x)
GAP_WID    = 0.001     # m  (grid-resolvable stand-in for the paper's 0.1 mm mold
                       #     gap between adjacent stacks; k scaled to compensate)
GPU_POWER  = 414.0     # W  (nominal 1.0f AI workload)
HBM_PER_STACK = 40.0   # W  per 12-Hi stack
TIERS      = 12        # DRAM dies per stack
EPS        = 1e-6      # passive-block placeholder power (W)

MAP_P2A    = 2.19      # cluster-map tile peak-to-average ratio (calibrated so
                       #     the baseline reproduces the paper's 141.7 C GPU peak)

FILLER_CENTER = "THERMAL_SI"  # paper baseline: thermal Si fills the central void
FILLER_GAP    = "MOLD"        # inter-stack gap (equivalent-k mold, see config)

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


def gpu_block(gpu_len, gpu_wid, label):
    """Single full-die block covering the whole die footprint (die-local)."""
    return [(f"{label}", 0.0, 0.0, gpu_len, gpu_wid, label)]


def hbm_tier_blocks(gpu_len, gpu_wid, hbm_side, gap_wid, macro_label):
    """Paper Fig. 3(a) baseline layout (die-local): 4 HBM stacks in the die
    corners (two per short edge), a mold gap of `gap_wid` between the two stacks
    on each edge (the un-merged baseline; 0 = merged), and the central
    THERMAL_SI column."""
    h = hbm_side
    central = gpu_len - 2 * h            # central column width (paper: 8 mm)
    sy = (gpu_wid - gap_wid) / 2.0       # stack extent in y
    xr = gpu_len - h                     # right stack-column X
    blocks = [
        ("HBM_BL", 0.0, 0.0,            h, sy, macro_label),
        ("HBM_TL", 0.0, sy + gap_wid,   h, sy, macro_label),
        ("HBM_BR", xr,  0.0,            h, sy, macro_label),
        ("HBM_TR", xr,  sy + gap_wid,   h, sy, macro_label),
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
    p = argparse.ArgumentParser(description="Generate PACT inputs for the BASELINE 3D HBM-on-GPU stack (unmerged stacks, non-uniform power, package surround).")
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--hbm-per-stack", type=float, default=HBM_PER_STACK)
    p.add_argument("--tiers", type=int, default=TIERS)
    p.add_argument("--gpu-len", type=float, default=GPU_LEN,
                   help="GPU die length (x, along the stack columns), m")
    p.add_argument("--gpu-wid", type=float, default=GPU_WID,
                   help="GPU die width (y, the short edges carrying the stacks), m")
    p.add_argument("--hbm-side", type=float, default=HBM_SIDE)
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
    p.add_argument("--no-base-die", action="store_true",
                   help="Remove the HBM base die (the paper's 'base die removal' study)")
    p.add_argument("--invert", action="store_true",
                   help="Inverted orientation: reverse the device stack so the GPU (BSPDN) sits "
                        "just under the TIM/lid (top cooling) and the HBM is buried below near "
                        "the adiabatic package side.")
    args = p.parse_args()

    if args.hbm_side * 2 >= args.gpu_len:
        raise SystemExit("Need 2*hbm-side < gpu-len so a central column separates the stacks.")
    if args.hbm_side * 2 > args.gpu_wid + 1e-12:
        raise SystemExit("Need 2*hbm-side <= gpu-wid so two stacks fit along each short edge.")
    gap = 0.0 if args.merged else args.gap_wid
    m = args.pkg_margin

    def ringed(blocks):
        return sc.add_package_ring(blocks, args.gpu_len, args.gpu_wid, m)

    print(f"\nGenerating BASELINE 3D HBM-on-GPU stack: "
          f"GPU {args.gpu_len*1e3:g} x {args.gpu_wid*1e3:g} mm @ {args.gpu_power:g} W "
          f"({args.power_map} power map), package "
          f"{(args.gpu_len+2*m)*1e3:g} x {(args.gpu_wid+2*m)*1e3:g} mm, "
          f"4 HBM stacks @ {args.hbm_per_stack:g} W ({args.tiers}-Hi"
          f"{'' if not args.no_base_die else ', base die removed'});"
          f" inter-stack gap = {gap*1e3:g} mm {FILLER_GAP if gap else '(merged)'}.\n")

    # ---- GPU + interface + package floorplans (die + PKG_MOLD ring) ----
    # every ptrace lists ALL blocks of its floorplan (ring blocks are passive)
    def passive_layer(flp_path, ptrace_path, blocks):
        write_flp(flp_path, blocks)
        write_ptrace(ptrace_path, [b[0] for b in blocks], {})

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

    # ---- HBM-stack sublayers (corner stacks + gaps + central Si + ring) ----
    base_beol = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, gap, "HBM_BASE_BEOL"))
    base_si   = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, gap, "HBM_BASE_SI"))
    hybond    = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, gap, "HYBRID_BOND"))
    dram_beol = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, gap, "DRAM_BEOL"))
    dram_si   = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, gap, "DRAM_SI"))
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
    # inverted orientation: reverse the device stack (BSPDN/GPU -> top, HBM -> bottom)
    if args.invert:
        layers = layers[::-1]
    # package cooling stack (always on top: the lid faces the GPU when inverted)
    layers.append(("tim_flp.csv", T_TIM, "tim_ptrace.csv"))
    layers.append(("lid_flp.csv", T_LID, "lid_ptrace.csv"))

    lcf_path = "dram3d_3d_lcf.csv"
    write_lcf(layers, lcf_path)

    total_mem = 4 * args.hbm_per_stack
    print(f"\n  Layers: {len(layers)} device layers"
          f"  (5 GPU/interface + {0 if args.no_base_die else 2} base die"
          f" + {3*args.tiers} DRAM sublayers + TIM + Lid)"
          f"  + NoPackage HTC boundary on the Lid")
    print(f"  Layout: central {FILLER_CENTER} column; inter-stack gap"
          f" {gap*1e3:g} mm {FILLER_GAP} (equivalent-k for the paper's 0.1 mm)"
          if gap else "  Layout: central THERMAL_SI column; stacks merged")
    print(f"  Ring  : {m*1e3:g} mm {sc.RING_LABEL} package surround, lid spans the package")
    print(f"  Power : GPU {args.gpu_power:g} W ({args.power_map}) + HBM {total_mem:g} W"
          f" = {args.gpu_power + total_mem:g} W total")
    print(f"  LCF   : {lcf_path}\n")


if __name__ == "__main__":
    main()
