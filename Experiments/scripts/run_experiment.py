"""
Driver for the PACT 3D-stack experiments.
=========================================
Given the path to an experiment directory, this:

  1. runs the experiment's input generator   (generate.py),
  2. runs the PACT steady-state solver        (../../src/PACT.py),
  3. renders all figures into  <experiment>/Results/.

Each experiment directory must contain a `meta.json` describing how to build and
plot it (generator command, LCF / config filenames, grid prefix, geometry, and
the list of plots). See Experiments.md for the format.

Usage:
    python3 run_experiment.py <experiment_dir> [--skip-gen] [--skip-sim] [--plots-only]

    <experiment_dir>   Path to an experiment (e.g. ../EXP_DRAM_R or ../EXP_NVDRAM_S).
    --skip-gen         Reuse existing generated inputs (don't re-run generate.py).
    --skip-sim         Reuse existing grid files (don't re-run PACT).
    --plots-only       Shortcut for --skip-gen --skip-sim (just redraw figures).

Tip: run EXP_DRAM_R before EXP_NVDRAM_S — the hybrid's peak-per-tier figure
compares against the pure-DRAM run.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))           # .../Experiments/scripts
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # .../PACT
PACT = os.path.join(REPO_ROOT, "src", "PACT.py")

sys.path.insert(0, HERE)
import pact_plots  # noqa: E402


def run(cmd, cwd):
    print(f"  $ {cmd}   (cwd={os.path.relpath(cwd, REPO_ROOT)})")
    subprocess.run(shlex.split(cmd), cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run a PACT 3D-stack experiment end to end.")
    ap.add_argument("experiment", help="Path to the experiment directory.")
    ap.add_argument("--skip-gen", action="store_true", help="Don't re-run generate.py.")
    ap.add_argument("--skip-sim", action="store_true", help="Don't re-run PACT.")
    ap.add_argument("--plots-only", action="store_true", help="= --skip-gen --skip-sim.")
    ap.add_argument("--no-compare", action="store_true",
                    help="Don't regenerate the cross-experiment comparison figures.")
    args = ap.parse_args()

    if args.plots_only:
        args.skip_gen = args.skip_sim = True

    exp_dir = os.path.abspath(args.experiment)
    meta_path = os.path.join(exp_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"No meta.json in {exp_dir}")
    with open(meta_path) as fh:
        meta = json.load(fh)

    print(f"\n=== {meta.get('name', os.path.basename(exp_dir))} :"
          f" {meta.get('title', '')} ===")

    # 1. generate inputs
    if not args.skip_gen:
        print("\n[1/3] Generating inputs ...")
        run(meta.get("generate", "python3 generate.py"), cwd=exp_dir)
    else:
        print("\n[1/3] Skipping input generation (--skip-gen).")

    # 2. run PACT (steady-state)
    if not args.skip_sim:
        print("\n[2/3] Running PACT (steady-state) ...")
        cmd = (f"python3 {shlex.quote(PACT)} {shlex.quote(meta['lcf'])} "
               f"{shlex.quote(meta['config'])} {shlex.quote(meta['modelparams'])} "
               f"--gridSteadyFile {shlex.quote(meta['grid_prefix'])}")
        run(cmd, cwd=exp_dir)
    else:
        print("\n[2/3] Skipping simulation (--skip-sim).")

    # 3. plots -> Results/
    print("\n[3/3] Rendering figures ...")
    out_dir = os.path.join(exp_dir, "Results")
    pact_plots.make_plots(exp_dir, meta, out_dir)
    print(f"\nDone. Figures in {os.path.relpath(out_dir, REPO_ROOT)}/")

    # 4. cross-experiment comparison figures (best-effort; needs >=2 experiments)
    if not args.no_compare:
        print("\n[4/4] Regenerating cross-experiment comparison figures ...")
        try:
            import comparisons
            exp_parent = os.path.dirname(exp_dir)
            comparisons.make_comparisons(exp_parent)
        except Exception as e:                       # never fail a run over comparisons
            print(f"  (comparison figures skipped: {e})")
    print()


if __name__ == "__main__":
    main()
