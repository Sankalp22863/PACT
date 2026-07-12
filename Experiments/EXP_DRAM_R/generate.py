"""
3D HBM-on-GPU Stack Generator  (paper replication, detailed layer stack)
========================================================================
Reproduces the thermal model from "Breaking Thermal Bottleneck in 3D HBM-on-GPU
Integration via System-Technology Co-Optimization": four 12-Hi HBM (DRAM) stacks
integrated *directly on top* of a GPU compute die, cooled by a top-side liquid
cold plate (the "lid"). The package side (bottom) is adiabatic, so heat is
forced *up* through the HBM stacks to the lid -- the "thermal bottleneck".

This generator expands every physical sublayer from the paper's cross-section
table (thicknesses + cross-plane thermal conductivity). Stack, bottom (adiabatic
package side, layer 0) -> top (lid / cold-plate side):

    BSPDN -> GPU FEOL (heat source, 414 W) -> BEOL_MXY -> Oxide
      -> GPU-HBM uBump
      -> HBM Base BEOL -> HBM Base Si (Base Die)
      -> 12 x [ Hybrid Bonding -> DRAM Die BEOL -> DRAM Die Si ]   (top die = 169 um)
      -> TIM -> Lid
    (PACT then applies the NoPackage convective HTC boundary on top of the Lid)

GPU / interface / TIM / Lid are full-die layers; the HBM-stack sublayers use the
paper's Fig. 3(a) die layout: a 30 x 22 mm GPU with the four 11 x 11 mm stacks
flush in the corners (two per short edge) and an 8 mm-wide thermal-silicon
column in the centre (the paper's vertical heat-escape path). All DRAM-die-Si
layers share `dram_tier_flp.csv` so downstream per-tier tools still find the 12
tiers; per-layer thickness is set in the LCF.

All layers are embedded in a molded package surround (PKG_MOLD ring + extended
copper lid, see scripts/stack_common.py) so heat can spread laterally beyond the
die as in the paper's 65x65 mm package.

Usage:
    python generate.py [--gpu-power W] [--hbm-per-stack W] [--tiers N]
                       [--gpu-len M] [--gpu-wid M] [--hbm-side M] [--no-base-die]
                       [--pkg-margin M]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import stack_common as sc  # noqa: E402

# ---- defaults (paper Fig. 3 die layout) ----
GPU_LEN    = 0.030     # m  (30 mm GPU die, long side = x; 11 + 8 + 11)
GPU_WID    = 0.022     # m  (22 mm GPU die, short side = y; 11 + 11)
HBM_SIDE   = 0.011     # m  (11 mm HBM stack footprint)
GPU_POWER  = 414.0     # W  (nominal 1.0f AI workload)
HBM_PER_STACK = 40.0   # W  per 12-Hi stack
TIERS      = 12        # DRAM dies per stack
GRID       = 48
EPS        = 1e-6      # passive-block placeholder power (W)

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
    """Single full-die block covering the whole footprint."""
    return [(f"{label}", 0.0, 0.0, gpu_len, gpu_wid, label)]


def hbm_tier_blocks(gpu_len, gpu_wid, hbm_side, macro_label, filler_label="THERMAL_SI"):
    """Paper Fig. 3(a) layout: 4 HBM stacks flush in the die corners (two per
    short edge) + a central filler column between them (8 mm thermal silicon in
    the paper -- the continuous high-k vertical escape path), so the bottleneck
    lives only inside the HBM stacks. If the die is wider than two stacks, thin
    filler strips pad the stack columns top/bottom. Returns (name, X, Y, L, W,
    Label) blocks tiling the die with no gaps/overlaps.
    """
    h = hbm_side
    central = gpu_len - 2 * h            # central column width (paper: 8 mm)
    my = (gpu_wid - 2 * h) / 2.0         # top/bottom margin (paper: 0)
    xr = gpu_len - h                     # right stack-column X
    blocks = [
        ("HBM_BL", 0.0, my,     h, h, macro_label),
        ("HBM_TL", 0.0, my + h, h, h, macro_label),
        ("HBM_BR", xr,  my,     h, h, macro_label),
        ("HBM_TR", xr,  my + h, h, h, macro_label),
        ("Center", h, 0.0, central, gpu_wid, filler_label),
    ]
    if my > 1e-9:
        blocks += [
            ("Strip_BL", 0.0, 0.0,        h, my, filler_label),
            ("Strip_TL", 0.0, my + 2 * h, h, my, filler_label),
            ("Strip_BR", xr,  0.0,        h, my, filler_label),
            ("Strip_TR", xr,  my + 2 * h, h, my, filler_label),
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
    p = argparse.ArgumentParser(description="Generate PACT inputs for a detailed 3D HBM-on-GPU stack.")
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--hbm-per-stack", type=float, default=HBM_PER_STACK)
    p.add_argument("--tiers", type=int, default=TIERS)
    p.add_argument("--gpu-len", type=float, default=GPU_LEN,
                   help="GPU die length (x, along the stack columns), m")
    p.add_argument("--gpu-wid", type=float, default=GPU_WID,
                   help="GPU die width (y, the short edges carrying the stacks), m")
    p.add_argument("--hbm-side", type=float, default=HBM_SIDE)
    p.add_argument("--pkg-margin", type=float, default=sc.PKG_MARGIN,
                   help="package mold ring width around the die (m); grid in modelParams must cover die+2*margin")
    p.add_argument("--grid", type=int, default=GRID)
    p.add_argument("--no-base-die", action="store_true",
                   help="Remove the HBM base die (the paper's 'base die removal' study)")
    args = p.parse_args()

    if args.hbm_side * 2 >= args.gpu_len:
        raise SystemExit("Need 2*hbm-side < gpu-len so a central column separates the stacks.")
    if args.hbm_side * 2 > args.gpu_wid + 1e-12:
        raise SystemExit("Need 2*hbm-side <= gpu-wid so two stacks fit along each short edge.")
    m = args.pkg_margin

    def ringed(blocks):
        return sc.add_package_ring(blocks, args.gpu_len, args.gpu_wid, m)

    def passive_layer(flp_path, ptrace_path, blocks):
        write_flp(flp_path, blocks)
        write_ptrace(ptrace_path, [b[0] for b in blocks], {})

    print(f"\nGenerating detailed 3D HBM-on-GPU stack: "
          f"GPU {args.gpu_len*1e3:g} x {args.gpu_wid*1e3:g} mm @ {args.gpu_power:g} W, package "
          f"{(args.gpu_len+2*m)*1e3:g} x {(args.gpu_wid+2*m)*1e3:g} mm, "
          f"4 HBM stacks @ {args.hbm_per_stack:g} W ({args.tiers}-Hi"
          f"{'' if not args.no_base_die else ', base die removed'}).\n")

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

    feol = ringed(gpu_block(args.gpu_len, args.gpu_wid, "GPU_FEOL"))
    write_flp("gpu_feol_flp.csv", feol)
    write_ptrace("gpu_feol_ptrace.csv", [b[0] for b in feol],
                 {"GPU_FEOL": args.gpu_power})

    # ---- HBM-stack sublayer floorplans (corner stacks + central Si + ring) ----
    base_beol = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, "HBM_BASE_BEOL"))
    base_si   = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, "HBM_BASE_SI"))
    hybond    = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, "HYBRID_BOND"))
    dram_beol = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, "DRAM_BEOL"))
    dram_si   = ringed(hbm_tier_blocks(args.gpu_len, args.gpu_wid, args.hbm_side, "DRAM_SI"))
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
    print(f"  Power : GPU {args.gpu_power:g} W + HBM {total_mem:g} W"
          f" = {args.gpu_power + total_mem:g} W total")
    print(f"  Die   : {args.gpu_len*1e3:g} x {args.gpu_wid*1e3:g} mm"
          f"  (grid rows/cols set in modelParams.config; 0.5 mm cells = paper resolution)")
    print(f"  LCF   : {lcf_path}\n")


if __name__ == "__main__":
    main()
