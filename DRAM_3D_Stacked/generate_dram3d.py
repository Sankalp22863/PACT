"""
3D HBM-on-GPU Stack Generator  (paper replication)
==================================================
Reproduces the thermal model from "Breaking Thermal Bottleneck in 3D HBM-on-GPU
Integration via System-Technology Co-Optimization": four 12-Hi HBM (DRAM) stacks
integrated *directly on top* of a GPU compute die, cooled by a top-side liquid
cold plate (the "lid"). The package/laminate side (bottom of the GPU) is the
weakly-cooled side.

Stack, bottom (laminate/package side, layer 0) -> top (lid / cold-plate side):

    GPU substrate/TSV -> GPU FEOL (heat source, 414 W) -> GPU BEOL
      -> uBump/hybrid-bond interface
      -> 12 x DRAM die        (each tier holds the 4 HBM stacks, side by side)
    (PACT then appends the NoPackage convective lid on top automatically)

PACT cools only from the top package; the bottom of layer 0 is adiabatic. That
is exactly the paper's regime (lid HTC 30,000 >> laminate HTC 200), so heat is
forced *up* through the HBM stacks to the lid -- the "thermal bottleneck".

Lateral layout (shared 48x48 grid on a `gpu_side` square, default 24 mm):
four `hbm_side` (10 mm) HBM stacks in a 2x2 arrangement; the cross-shaped gap
between them and the outer frame are filled with high-conductivity "thermal
silicon" (140 W/m.K) -- the paper's targeted vertical heat-escape paths.

  2.5D reference mode (--mode 2p5d): the HBM is NOT stacked on the GPU (it sits
  beside it on an interposer), so the GPU die is cooled directly by the lid with
  no memory loading its vertical path. Generates only the 3 GPU layers.

Usage:
    python generate_dram3d.py [--mode 3d|2p5d] [--gpu-power W] [--hbm-per-stack W]
                              [--tiers N] [--gpu-side M] [--hbm-side M] [--grid N]

    --mode          3d (HBM stacked on GPU, default) or 2p5d (GPU-only reference)
    --gpu-power     GPU active power, W (default 414; paper f-scaling: 391/368/346/323/300)
    --hbm-per-stack power per 12-Hi HBM stack, W (default 40)
    --tiers         DRAM dies per HBM stack (default 12 = "12-Hi")
    --gpu-side      GPU die edge, m (default 0.024)
    --hbm-side      HBM stack edge, m (default 0.010)
    --grid          grid cells per side (default 48)
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

# Layer thicknesses (m)
T_GPU_SUB  = 100e-6
T_GPU_FEOL = 2e-6
T_GPU_BEOL = 8e-6
T_UBUMP    = 15e-6
T_DRAM     = 50e-6     # per thinned DRAM die (incl. its thin BEOL)
T_DAF      = 5e-6      # die-attach film between stacked DRAM dies (the bottleneck)


def mm(x):
    return x  # coords are kept in metres throughout; helper kept for clarity


def gpu_block(gpu_side, label):
    """Single full-die block covering the whole GPU footprint."""
    return [(f"{label}", 0.0, 0.0, gpu_side, gpu_side, label)]


def hbm_tier_blocks(gpu_side, hbm_side, macro_label="DRAM_Si"):
    """4 HBM stacks in 2x2 + thermal-silicon cross/frame filling the rest.

    macro_label selects what the four stack footprints are made of:
      "DRAM_Si" for a DRAM-die layer, or "DAF" for an inter-die die-attach film.
    The cross/frame is always THERMAL_SI (continuous high-k escape path), so the
    bottleneck lives only inside the HBM stacks.

    Returns list of (name, X, Y, Length, Width, Label). Blocks tile the die with
    no gaps/overlaps; coordinates land on grid lines for the default geometry.
    """
    g = gpu_side
    h = hbm_side
    # symmetric margins and central gap so 2*margin + 2*h + gap == g
    gap = g - 2 * h                      # total slack split as: margin | h | central | h | margin
    margin = gap / 4.0                   # outer frame width
    central = gap / 2.0                  # central cross width
    a0 = margin                          # left/bottom stack start
    a1 = margin + h                      # left/bottom stack end / central start
    a2 = margin + h + central            # right/top stack start
    a3 = margin + h + central + h        # right/top stack end (== g - margin)

    blocks = []
    # --- 4 HBM stacks ---
    blocks.append(("HBM_BL", a0, a0, h, h, macro_label))   # bottom-left
    blocks.append(("HBM_BR", a2, a0, h, h, macro_label))   # bottom-right
    blocks.append(("HBM_TL", a0, a2, h, h, macro_label))   # top-left
    blocks.append(("HBM_TR", a2, a2, h, h, macro_label))   # top-right
    # --- outer frame (thermal silicon) ---
    blocks.append(("Frame_B", 0.0, 0.0, g, margin, "THERMAL_SI"))            # bottom strip
    blocks.append(("Frame_T", 0.0, a3, g, margin, "THERMAL_SI"))            # top strip
    blocks.append(("Frame_L", 0.0, margin, margin, g - 2 * margin, "THERMAL_SI"))  # left
    blocks.append(("Frame_R", a3, margin, margin, g - 2 * margin, "THERMAL_SI"))   # right
    # --- central cross (thermal silicon) ---
    blocks.append(("Cross_V", a1, margin, central, g - 2 * margin, "THERMAL_SI"))  # vertical bar
    blocks.append(("Cross_HL", a0, a1, h, central, "THERMAL_SI"))                  # horiz left seg
    blocks.append(("Cross_HR", a2, a1, h, central, "THERMAL_SI"))                  # horiz right seg
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
    p = argparse.ArgumentParser(description="Generate PACT inputs for a 3D HBM-on-GPU stack.")
    p.add_argument("--mode", choices=["3d", "2p5d"], default="3d")
    p.add_argument("--gpu-power", type=float, default=GPU_POWER)
    p.add_argument("--hbm-per-stack", type=float, default=HBM_PER_STACK)
    p.add_argument("--tiers", type=int, default=TIERS)
    p.add_argument("--gpu-side", type=float, default=GPU_SIDE)
    p.add_argument("--hbm-side", type=float, default=HBM_SIDE)
    p.add_argument("--grid", type=int, default=GRID)
    args = p.parse_args()

    if args.mode == "3d" and args.hbm_side * 2 >= args.gpu_side:
        raise SystemExit("Need 2*hbm-side < gpu-side so the 2x2 HBM array fits with margins.")

    print(f"\nGenerating 3D HBM-on-GPU stack: mode={args.mode}, "
          f"GPU {args.gpu_side*1e3:g} mm @ {args.gpu_power:g} W, "
          f"{4 if args.mode=='3d' else 0} HBM stacks @ {args.hbm_per_stack:g} W "
          f"({args.tiers}-Hi).\n")

    # ---- GPU layer floorplans (single full-die block each) ----
    write_flp("gpu_substrate_flp.csv", gpu_block(args.gpu_side, "GPU_Si"))
    write_flp("gpu_feol_flp.csv",      gpu_block(args.gpu_side, "GPU_FEOL"))
    write_flp("gpu_beol_flp.csv",      gpu_block(args.gpu_side, "GPU_BEOL"))

    # ---- GPU ptraces ----
    write_ptrace("gpu_substrate_ptrace.csv", ["GPU_Si"], {})
    write_ptrace("gpu_feol_ptrace.csv",      ["GPU_FEOL"], {"GPU_FEOL": args.gpu_power})
    write_ptrace("gpu_beol_ptrace.csv",      ["GPU_BEOL"], {})

    layers = [
        ("gpu_substrate_flp.csv", T_GPU_SUB,  "gpu_substrate_ptrace.csv"),
        ("gpu_feol_flp.csv",      T_GPU_FEOL, "gpu_feol_ptrace.csv"),
        ("gpu_beol_flp.csv",      T_GPU_BEOL, "gpu_beol_ptrace.csv"),
    ]

    if args.mode == "3d":
        # interface + HBM tiers
        write_flp("ubump_flp.csv", gpu_block(args.gpu_side, "UBUMP"))
        write_ptrace("ubump_ptrace.csv", ["UBUMP"], {})
        layers.append(("ubump_flp.csv", T_UBUMP, "ubump_ptrace.csv"))

        # DRAM-die layer: HBM macros = active DRAM_Si; cross/frame = thermal Si.
        die_blocks = hbm_tier_blocks(args.gpu_side, args.hbm_side, macro_label="DRAM_Si")
        names = [b[0] for b in die_blocks]
        write_flp("dram_tier_flp.csv", die_blocks)
        per_die_per_stack = args.hbm_per_stack / args.tiers
        pmap = {n: per_die_per_stack for n in names if n.startswith("HBM_")}
        write_ptrace("dram_tier_ptrace.csv", names, pmap)

        # Die-attach-film layer: HBM macros = low-k DAF bond; cross/frame = thermal
        # Si (so the escape paths stay continuous). Passive.
        daf_blocks = hbm_tier_blocks(args.gpu_side, args.hbm_side, macro_label="DAF")
        write_flp("daf_flp.csv", daf_blocks)
        write_ptrace("daf_ptrace.csv", names, {})

        # Interleave: die, DAF, die, DAF, ... (N dies, N-1 bond films between them).
        for t in range(args.tiers):
            layers.append(("dram_tier_flp.csv", T_DRAM, "dram_tier_ptrace.csv"))
            if t < args.tiers - 1:
                layers.append(("daf_flp.csv", T_DAF, "daf_ptrace.csv"))

    lcf_path = f"dram3d_{args.mode}_lcf.csv"
    write_lcf(layers, lcf_path)

    # ---- summary ----
    n_dram_layers = args.tiers if args.mode == "3d" else 0
    total_mem = 4 * args.hbm_per_stack if args.mode == "3d" else 0.0
    print(f"\n  Layers: {len(layers)} device layers"
          f" ({3} GPU + {1 if args.mode=='3d' else 0} interface + {n_dram_layers} DRAM)"
          f"  + 1 NoPackage lid (added by PACT)")
    print(f"  Power : GPU {args.gpu_power:g} W + HBM {total_mem:g} W"
          f" = {args.gpu_power + total_mem:g} W total")
    print(f"  Grid  : {args.grid}x{args.grid} on {args.gpu_side*1e3:g} mm die"
          f"  ({args.gpu_side/args.grid*1e3:.3f} mm/cell)")
    print(f"  LCF   : {lcf_path}\n")


if __name__ == "__main__":
    main()
