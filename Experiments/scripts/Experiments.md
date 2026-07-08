# PACT 3D-Stack Experiments — Overview

Thermal experiments for 3D memory-on-GPU integration, built on the PACT
SPICE/SuperLU thermal simulator (`../../src/PACT.py`). All experiments share the
same compute-die-on-bottom, liquid-lid-on-top regime from the IEDM paper
*"Breaking Thermal Bottleneck in 3D HBM-on-GPU Integration via System-Technology
Co-Optimization"* — heat is forced **up** from the GPU through the memory stack
to a top cold-plate lid (the package side is adiabatic).

```
Experiments/
├── EXP_DRAM_R/        # DRAM (HBM) on a 3D-stacked compute die — paper reproduction
├── EXP_NVDRAM_S/      # N stacked NVDRAM below DRAM on the compute die
└── scripts/           # shared driver + plotting (this folder)
```

## The two experiments

| Dir | What it models |
|---|---|
| **EXP_DRAM_R** | Four 12-Hi HBM (DRAM) stacks directly on a GPU compute die, in a 2×2 layout with a central thermal-silicon escape column. Reproduces the paper's HBM-on-GPU thermal result. |
| **EXP_NVDRAM_S** | A hybrid stack: **N NVDRAM tiers at the bottom** (nearest the GPU) with the **DRAM (HBM) tiers continuing on top**, same 2×2 + thermal-silicon geometry. Studies the thermal cost of a low-conductivity non-volatile memory in the vertical heat-escape path. |

Each experiment folder has its own `README.md` with the full stack, materials,
and power model.

## How to run

The driver takes an experiment path, runs its generator, runs PACT, and writes
every figure into `<experiment>/Results/`:

```bash
cd Experiments/scripts
python3 run_experiment.py ../EXP_DRAM_R      # run this first
python3 run_experiment.py ../EXP_NVDRAM_S    # uses EXP_DRAM_R for the comparison plot
```

Flags: `--skip-gen` (reuse generated inputs), `--skip-sim` (reuse grids),
`--plots-only` (just redraw figures). Run **EXP_DRAM_R before EXP_NVDRAM_S** — the
hybrid's `peak_per_tier` figure compares against the pure-DRAM run.

## Figures produced (in `<experiment>/Results/`)

| File | Description |
|---|---|
| `heatmaps.png` | Per-layer steady-state temperature heatmaps, shared color scale. |
| `lowest_layer_surface.png` | Lowest layer as a tilted flat color plane (color = temperature, peak annotated). |
| `cross_section.png` | Stack cross-section schematic (memory \| thermal-silicon \| memory). |
| `layer_table.png` | Per-layer thickness + in-plane / cross-plane thermal conductivity. |
| `future_integration.png` | High-level "future 3D integration" block diagram. |
| `peak_per_tier_hybrid_vs_dram.png` | *(EXP_NVDRAM_S only)* peak temperature up the stack: hybrid vs pure DRAM. |

## `meta.json` format

Each experiment directory carries a `meta.json` that tells the driver how to
build and plot it:

```json
{
  "name": "EXP_DRAM_R",
  "title": "...",
  "generate": "python3 generate.py",     // command that writes flp/ptrace/lcf
  "lcf": "dram3d_3d_lcf.csv",            // layer config file (generator output)
  "config": "experiment.config",         // material thermal properties
  "modelparams": "modelParams.config",   // solver + grid + lid settings
  "grid_prefix": "dram3d_3d.grid.steady",// PACT --gridSteadyFile prefix
  "gpu_side": 0.024, "mem_side": 0.010,  // geometry (m), for heatmap overlays
  "plots": ["heatmaps", "surface", "cross_section", "table", "future"],
  "compare_dram": {                       // EXP_NVDRAM_S only, for peak_per_tier
    "dir": "../EXP_DRAM_R",
    "grid_prefix": "dram3d_3d.grid.steady",
    "lcf": "dram3d_3d_lcf.csv"
  }
}
```

## Files

| In `scripts/` | Purpose |
|---|---|
| `run_experiment.py` | Driver: generate → PACT → figures into `Results/`. |
| `pact_plots.py` | Shared plotting library (all figure types). |
| `Experiments.md` | This overview. |

Generated inputs (`*_flp.csv`, `*_ptrace.csv`, `*_lcf.csv`), raw grid outputs
(`*.grid.steady.*`), and the `Results/` figures are **not** tracked in git — they
are reproduced by re-running the driver.
