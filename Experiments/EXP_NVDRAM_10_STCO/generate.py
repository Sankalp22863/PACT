"""
STCO Waterfall Generator — RECTANGULAR die  (NVDRAM_baseline2)
=============================================================
Rectangular-die variant of the waterfall generator: the GPU die is
gpu_x × gpu_y (default 30 × 22 mm, matching the imec IEDM 2025 Fig. 3 layout)
instead of the square 24 × 24 mm used by the other experiments. Everything else
(materials, per-stage interventions, power model) is unchanged. This folder is
used for the ALL-DRAM baseline (--nv-tiers 0 --dram-tiers 12).

Grid: PACT derives grid_length = length/cols and grid_width = width/rows
independently, so a rectangular die needs a matching rectangular grid. With
0.5 mm cells, 30 × 22 mm -> cols = 60, rows = 44 (set in modelParams.config).

Interventions (per the imec STCO staircase):
  * --no-base-die   : HBM base-die removal
  * --merge-stacks  : replace inter-stack mold cross with continuous silicon (MERGE_SI, k=140)
  * --thin-top-die  : top memory die Si 169 um -> 50 um
  * --gpu-power W    : GPU frequency scaling (linear P~f)
  * --optimized      : thermal-silicon optimization (filler MOLD -> THERMAL_SI)

Usage:
    python3 waterfall_generate.py [--nv-tiers N] [--dram-tiers N] [--gpu-power W]
        [--gpu-x M] [--gpu-y M] [--mem-side M] [--dram-per-stack W]
        [--dram-refresh-frac F] [--nv-leakage-factor F]
        [--no-base-die] [--merge-stacks] [--thin-top-die] [--optimized]
"""

import argparse

# ---- defaults ----
GPU_X      = 0.030     # m  (30 mm GPU die width,  imec Fig. 3)
GPU_Y      = 0.022     # m  (22 mm GPU die height, imec Fig. 3)
MEM_SIDE   = 0.010     # m  (10 mm memory-stack footprint; must fit 2x2 with margins)
GPU_POWER  = 414.0     # W
DRAM_PER_STACK = 40.0  # W  per 12-Hi stack
NV_TIERS   = 0         # all-DRAM baseline by default
DRAM_TIERS = 12
EPS        = 1e-6

FILLER_BASELINE  = "MOLD"        # mold compound, k = 3 W/m-K (paper value)
FILLER_OPTIMIZED = "THERMAL_SI"  # high-k dummy silicon, k = 140
FILLER_MERGE     = "MERGE_SI"    # continuous silicon bridge, k = 140 (stack merging)

# ---- paper (imec Fig. 3) 3D layout: 2 stack columns on the short edges ----
EDGE_INSERT  = 0.001    # m, 1 mm thermal-silicon-insert strip at each outer package edge
CENTRAL_VOID = 0.008    # m, 8 mm central void between the two stack columns

DRAM_REFRESH_FRAC = 0.35
NV_LEAKAGE_FACTOR = 0.5

# ---- layer thicknesses (m) ----
T_BSPDN, T_GPU_FEOL, T_BEOL_MXY, T_OXIDE = 1.715e-6, 0.15e-6, 1.4e-6, 1.0e-6
T_GPU_SI_DEFAULT = 100e-6   # GPU bulk-Si substrate (thinned die); enables lateral spreading
T_UBUMP, T_BASE_BEOL, T_BASE_SI = 40e-6, 5e-6, 50e-6
T_HYBOND, T_DIE_BEOL, T_DIE_SI, T_DRAM_TOP = 2e-6, 3e-6, 50e-6, 169e-6
T_TIM, T_LID = 200e-6, 3000e-6


def gpu_block(gx, gy, label):
    return [(label, 0.0, 0.0, gx, gy, label)]


def mem_tier_blocks(gx, gy, h, macro_label, filler_label, merge=False):
    """Paper (imec Fig. 3) 3D layout on a gx x gy die:

      x: [0,EDGE] insert | [.., cw] LEFT stack column | [.., VOID] central void |
         [.., cw] RIGHT stack column | [.., EDGE] insert
      y: two stacks per column, [0, gy/2] and [gy/2, gy]  (they reach both edges)

    Two HBM stack columns line the left/right SHORT edges (each column = 2 stacks
    stacked vertically); a central VOID sits between them; thin thermal-silicon
    inserts sit at the two outer package edges. There is NO perimeter mold frame.

      * Central void = MERGE_SI (k=140) when --merge-stacks, else `filler_label`
        (MOLD at baseline)  -> the paper's "replace the intervening mold between
        adjacent stacks with a continuous silicon region" (the big merge lever).
      * Edge inserts = `filler_label` (MOLD -> THERMAL_SI when --optimized)
        -> the paper's "narrow thermal-silicon inserts at the package edges".

    So the baseline hotspot is the CENTRAL mold void (like the paper), which merge
    converts to silicon. `h` (mem_side) is unused in this layout.
    """
    cw = (gx - 2 * EDGE_INSERT - CENTRAL_VOID) / 2.0     # stack-column width
    x0 = EDGE_INSERT
    x1 = EDGE_INSERT + cw                                 # void start
    x2 = EDGE_INSERT + cw + CENTRAL_VOID                  # right column start
    x3 = EDGE_INSERT + cw + CENTRAL_VOID + cw             # right edge start (= gx - EDGE)
    ymid = gy / 2.0
    void_label = FILLER_MERGE if merge else filler_label
    blocks = [
        ("MEM_LB", x0, 0.0,  cw, ymid, macro_label),     # left-bottom stack
        ("MEM_LT", x0, ymid, cw, ymid, macro_label),     # left-top
        ("MEM_RB", x2, 0.0,  cw, ymid, macro_label),     # right-bottom
        ("MEM_RT", x2, ymid, cw, ymid, macro_label),     # right-top
        ("Void",   x1, 0.0,  CENTRAL_VOID, gy, void_label),   # central void (merge region)
        ("Edge_L", 0.0, 0.0, EDGE_INSERT, gy, filler_label),  # left package-edge insert
        ("Edge_R", x3,  0.0, EDGE_INSERT, gy, filler_label),  # right package-edge insert
    ]
    return blocks, ["MEM_LB", "MEM_LT", "MEM_RB", "MEM_RT"]


def write_flp(path, blocks):
    lines = ["UnitName,X,Y,Length (m),Width (m),ConfigFile,Label"]
    for name, x, y, l, w, label in blocks:
        lines.append(f"{name},{x:g},{y:g},{l:g},{w:g},,{label}")
    open(path, "w").write("\n".join(lines) + "\n")


def write_ptrace(path, names, power_map):
    lines = ["UnitName,Power,Power1,Power2"]
    for n in names:
        p = power_map.get(n, EPS)
        lines.append(f"{n},{p:g},{p:g},{p:g}")
    open(path, "w").write("\n".join(lines) + "\n")


def write_lcf(layers, path):
    lines = ["Layer,FloorplanFile,Thickness (m),PtraceFile,LateralHeatFlow"]
    for idx, (flp, thickness, ptrace) in enumerate(layers):
        lines.append(f"{idx},{flp},{thickness:.3e},{ptrace},True")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    global CENTRAL_VOID
    p = argparse.ArgumentParser(description="Generate PACT inputs for one rectangular-die STCO waterfall stage.")
    p.add_argument("--nv-tiers", type=int, default=NV_TIERS)
    p.add_argument("--dram-tiers", type=int, default=DRAM_TIERS)
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--gpu-x", type=float, default=GPU_X)
    p.add_argument("--gpu-y", type=float, default=GPU_Y)
    p.add_argument("--mem-side", type=float, default=MEM_SIDE)
    p.add_argument("--dram-per-stack", type=float, default=DRAM_PER_STACK)
    p.add_argument("--dram-refresh-frac", type=float, default=DRAM_REFRESH_FRAC)
    p.add_argument("--nv-leakage-factor", type=float, default=NV_LEAKAGE_FACTOR)
    p.add_argument("--gpu-si-um", type=float, default=T_GPU_SI_DEFAULT * 1e6,
                   help="GPU bulk-Si substrate thickness in um (0 = none). Enables lateral spreading.")
    p.add_argument("--void-um", type=float, default=CENTRAL_VOID * 1e6,
                   help="effective mold-void width in um that merge converts (calibrates the merge step)")
    p.add_argument("--no-base-die", action="store_true")
    p.add_argument("--merge-stacks", action="store_true")
    p.add_argument("--thin-top-die", action="store_true")
    p.add_argument("--optimized", action="store_true")
    # active read/write model (baseline3/waterfall3): add access dynamic power on top of standby
    p.add_argument("--active", action="store_true",
                   help="add active read/write dynamic power on top of the standby/refresh power")
    p.add_argument("--stack-bandwidth-tbps", type=float, default=4.4,
                   help="peak per-stack HBM bandwidth in TB/s (fully-stacked model, not 2.5D)")
    p.add_argument("--mem-activity-factor", type=float, default=0.2421,
                   help="average memory activity factor alpha_w (weight-traffic fraction of HBM BW)")
    p.add_argument("--dram-rw-energy-fj", type=float, default=70.0,
                   help="DRAM read/write energy in fJ/bit")
    p.add_argument("--nv-rw-energy-fj", type=float, default=90.0,
                   help="NVDRAM read/write energy in fJ/bit")
    args = p.parse_args()
    CENTRAL_VOID = args.void_um * 1e-6     # tunable merge-void width (stacks widen to fill)

    gx, gy, h = args.gpu_x, args.gpu_y, args.mem_side
    if 2 * h >= gx or 2 * h >= gy:
        raise SystemExit("Need 2*mem-side < gpu-x and < gpu-y so the 2x2 array fits with margins.")

    filler = FILLER_OPTIMIZED if args.optimized else FILLER_BASELINE
    top_si = T_DIE_SI if args.thin_top_die else T_DRAM_TOP
    merge = args.merge_stacks

    # full-die GPU + interface + package floorplans
    write_flp("bspdn_flp.csv",    gpu_block(gx, gy, "BSPDN"))
    write_flp("gpu_si_flp.csv",   gpu_block(gx, gy, "GPU_SI"))
    write_flp("gpu_feol_flp.csv", gpu_block(gx, gy, "GPU_FEOL"))
    write_flp("beol_mxy_flp.csv", gpu_block(gx, gy, "BEOL_MXY"))
    write_flp("oxide_flp.csv",    gpu_block(gx, gy, "OXIDE"))
    write_flp("ubump_flp.csv",    gpu_block(gx, gy, "GPU_HBM_UBUMP"))
    write_flp("tim_flp.csv",      gpu_block(gx, gy, "TIM"))
    write_flp("lid_flp.csv",      gpu_block(gx, gy, "LID"))
    write_ptrace("bspdn_ptrace.csv",    ["BSPDN"], {})
    write_ptrace("gpu_si_ptrace.csv",   ["GPU_SI"], {})
    write_ptrace("gpu_feol_ptrace.csv", ["GPU_FEOL"], {"GPU_FEOL": args.gpu_power})
    write_ptrace("beol_mxy_ptrace.csv", ["BEOL_MXY"], {})
    write_ptrace("oxide_ptrace.csv",    ["OXIDE"], {})
    write_ptrace("ubump_ptrace.csv",    ["GPU_HBM_UBUMP"], {})
    write_ptrace("tim_ptrace.csv",      ["TIM"], {})
    write_ptrace("lid_ptrace.csv",      ["LID"], {})

    # memory-stack sublayer floorplans
    base_beol, _  = mem_tier_blocks(gx, gy, h, "HBM_BASE_BEOL", filler, merge)
    base_si, _    = mem_tier_blocks(gx, gy, h, "HBM_BASE_SI",   filler, merge)
    hybond, _     = mem_tier_blocks(gx, gy, h, "HYBRID_BOND",   filler, merge)
    nv_beol, _    = mem_tier_blocks(gx, gy, h, "NV_DIE_BEOL",   filler, merge)
    nv_si, mem_nm = mem_tier_blocks(gx, gy, h, "NV_DIE_SI",     filler, merge)
    dram_beol, _  = mem_tier_blocks(gx, gy, h, "DRAM_BEOL",     filler, merge)
    dram_si, _    = mem_tier_blocks(gx, gy, h, "DRAM_SI",       filler, merge)
    all_names = [b[0] for b in dram_si]
    write_flp("hbm_base_beol_flp.csv", base_beol)
    write_flp("hbm_base_si_flp.csv",   base_si)
    write_flp("hybrid_bond_flp.csv",   hybond)
    write_flp("nv_die_beol_flp.csv",   nv_beol)
    write_flp("nv_tier_flp.csv",       nv_si)
    write_flp("dram_die_beol_flp.csv", dram_beol)
    write_flp("dram_tier_flp.csv",     dram_si)

    total_tiers  = args.nv_tiers + args.dram_tiers
    dram_standby = args.dram_per_stack / total_tiers
    nv_standby   = args.nv_leakage_factor * dram_standby * (1.0 - args.dram_refresh_frac)
    # --- active read/write dynamic power (baseline3/waterfall3) ---
    #   P_active = alpha_w * (per-die share of peak stack bandwidth) * E_rw(technology)
    # The per-stack peak bandwidth is shared uniformly across the memory dies, and each
    # die dissipates its share at its own read/write energy (70 fJ/bit DRAM, 90 fJ/bit NVDRAM).
    dram_active = nv_active = 0.0
    if args.active:
        stack_bits_per_s = args.stack_bandwidth_tbps * 1e12 * 8.0        # TB/s -> bit/s
        bits_per_s_per_die = args.mem_activity_factor * stack_bits_per_s / total_tiers
        dram_active = bits_per_s_per_die * args.dram_rw_energy_fj * 1e-15   # fJ/bit -> J/bit
        nv_active   = bits_per_s_per_die * args.nv_rw_energy_fj   * 1e-15
    dram_per_die = dram_standby + dram_active
    nv_per_die   = nv_standby   + nv_active
    write_ptrace("nv_tier_ptrace.csv",   all_names, {n: nv_per_die   for n in mem_nm})
    write_ptrace("dram_tier_ptrace.csv", all_names, {n: dram_per_die for n in mem_nm})
    write_ptrace("mem_passive_ptrace.csv", all_names, {})

    t_gpu_si = args.gpu_si_um * 1e-6
    layers = [("bspdn_flp.csv", T_BSPDN, "bspdn_ptrace.csv")]
    if t_gpu_si > 0:                       # thick GPU Si substrate (below FEOL) for lateral spreading
        layers.append(("gpu_si_flp.csv", t_gpu_si, "gpu_si_ptrace.csv"))
    layers += [
        ("gpu_feol_flp.csv", T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("beol_mxy_flp.csv", T_BEOL_MXY, "beol_mxy_ptrace.csv"),
        ("oxide_flp.csv",    T_OXIDE,    "oxide_ptrace.csv"),
        ("ubump_flp.csv",    T_UBUMP,    "ubump_ptrace.csv"),
    ]
    if not args.no_base_die:
        layers.append(("hbm_base_beol_flp.csv", T_BASE_BEOL, "mem_passive_ptrace.csv"))
        layers.append(("hbm_base_si_flp.csv",   T_BASE_SI,   "mem_passive_ptrace.csv"))
    for i in range(args.nv_tiers):
        nv_is_top = (args.dram_tiers == 0 and i == args.nv_tiers - 1)
        t_si = top_si if nv_is_top else T_DIE_SI
        layers += [("hybrid_bond_flp.csv", T_HYBOND, "mem_passive_ptrace.csv"),
                   ("nv_die_beol_flp.csv", T_DIE_BEOL, "mem_passive_ptrace.csv"),
                   ("nv_tier_flp.csv", t_si, "nv_tier_ptrace.csv")]
    for t in range(args.dram_tiers):
        t_si = top_si if t == args.dram_tiers - 1 else T_DIE_SI
        layers += [("hybrid_bond_flp.csv", T_HYBOND, "mem_passive_ptrace.csv"),
                   ("dram_die_beol_flp.csv", T_DIE_BEOL, "mem_passive_ptrace.csv"),
                   ("dram_tier_flp.csv", t_si, "dram_tier_ptrace.csv")]
    layers += [("tim_flp.csv", T_TIM, "tim_ptrace.csv"),
               ("lid_flp.csv", T_LID, "lid_ptrace.csv")]
    write_lcf(layers, "hybrid_lcf.csv")

    ints = [s for s, on in [("base-die-removal", args.no_base_die), ("stack-merging", merge),
                            ("top-die-thinning", args.thin_top_die), ("thermal-silicon", args.optimized)] if on]
    act = (f" | active: +{dram_active*1e3:.1f} mW/DRAM-die, +{nv_active*1e3:.1f} mW/NV-die "
           f"(a_w={args.mem_activity_factor:g}, {args.stack_bandwidth_tbps:g} TB/s/stack, "
           f"{args.dram_rw_energy_fj:g}/{args.nv_rw_energy_fj:g} fJ/bit)") if args.active else " | standby-only"
    print(f"[waterfall_generate rect] die {gx*1e3:g}x{gy*1e3:g} mm | {len(layers)} layers | "
          f"filler={filler} | gpu={args.gpu_power:g} W | "
          f"{args.nv_tiers} NVDRAM + {args.dram_tiers} DRAM | "
          f"DRAM {dram_per_die:.3f} W/die, NV {nv_per_die:.3f} W/die{act} | "
          f"interventions={ints if ints else ['none (baseline)']}")


if __name__ == "__main__":
    main()
