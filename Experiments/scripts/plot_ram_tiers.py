"""
CLI for the cross-experiment comparison figures.
=================================================
Thin wrapper around comparisons.py. Parses the NVDRAM/DRAM stack-composition
experiments and renders:

  * ram_tier_temperatures.png - peak temperature of each memory die up the stack
  * peak_vs_nv_fraction.png   - GPU vs memory peak (and chip power) vs NVDRAM count

Usage:
    python3 plot_ram_tiers.py [exp_dir ...] [--baseline DIR] [--out-dir DIR]
                              [--no-baseline]

    exp_dir        Experiment dirs (default: all ../EXP_NVDRAM_<n>, by NV count).
    --baseline     Pure-DRAM reference experiment (default: ../EXP_DRAM_R).
    --no-baseline  Don't include the pure-DRAM baseline.
    --out-dir      Output directory (default: ../comparisons).
"""

import argparse
import os

import comparisons as C


def main():
    ap = argparse.ArgumentParser(description="Render cross-experiment memory-tier "
                                             "and GPU-vs-memory comparison charts.")
    ap.add_argument("experiments", nargs="*", help="Experiment dirs (default: all EXP_NVDRAM_<n>).")
    ap.add_argument("--baseline", default=os.path.join(C.EXP_ROOT, "EXP_DRAM_R"),
                    help="Pure-DRAM reference experiment dir.")
    ap.add_argument("--no-baseline", action="store_true", help="Skip the pure-DRAM baseline.")
    ap.add_argument("--out-dir", default=os.path.join(C.EXP_ROOT, "comparisons"))
    args = ap.parse_args()

    if args.experiments:
        profiles = [p for p in (C.tier_profile(os.path.abspath(d)) for d in args.experiments) if p]
        profiles.sort(key=lambda p: p["n_nv"])
        if not args.no_baseline and args.baseline:
            b = C.tier_profile(os.path.abspath(args.baseline))
            if b and b["dir"] not in {p["dir"] for p in profiles}:
                profiles.insert(0, b)
    else:
        profiles = C.discover_profiles(C.EXP_ROOT, baseline=None if args.no_baseline
                                       else os.path.basename(args.baseline))

    print(f"\nPlotting {len(profiles)} experiment(s):")
    for p in profiles:
        print(f"  {p['name']:16s}  {C.series_label(p)}  "
              f"(GPU {p['gpu']:.1f}°C, mem {p['tiers'][0][1]:.1f}→{p['tiers'][-1][1]:.1f}°C, "
              f"total {p['total_power']:.0f} W)")
    if len(profiles) < 2:
        raise SystemExit("Need >=2 experiments with results to compare.")

    out_dir = os.path.abspath(args.out_dir)
    C.ram_tier_temps(profiles, os.path.join(out_dir, "ram_tier_temperatures.png"))
    C.peak_vs_nv_fraction(profiles, os.path.join(out_dir, "peak_vs_nv_fraction.png"))


if __name__ == "__main__":
    main()
