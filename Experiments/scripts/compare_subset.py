"""
Render the cross-experiment RAM-tier temperature figure for an arbitrary
subset of experiments (same plot as comparisons.py's ram_tier_temperatures.png
but with hand-picked experiment directories and an explicit output name).

Usage (from Experiments/scripts):
    python3 compare_subset.py ../EXP_NVDRAM_2_STCO ../EXP_NVDRAM_4_STCO \
        ../EXP_NVDRAM_6_STCO -o ../comparisons/ram_tier_temperatures_comp.png
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comparisons  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="RAM-tier temperature comparison for chosen experiments.")
    ap.add_argument("experiments", nargs="+", help="experiment directories to compare (in plot order)")
    ap.add_argument("-o", "--out", required=True, help="output PNG path")
    args = ap.parse_args()

    profiles = []
    for d in args.experiments:
        p = comparisons.tier_profile(os.path.abspath(d))
        if p is None:
            raise SystemExit(f"no usable results in {d} (run the experiment first)")
        profiles.append(p)
    print("  Comparing: " + ", ".join(comparisons.series_label(p) for p in profiles))
    comparisons.ram_tier_temps(profiles, os.path.abspath(args.out))


if __name__ == "__main__":
    main()
