# NVDRAM_waterfall3 — φ-HBM STCO waterfall **with active read/write** (30 × 22 mm)

The active-access counterpart of [`../NVDRAM_waterfall2`](../NVDRAM_waterfall2/):
identical paper-faithful geometry and calibration, φ-HBM stack
(**4 NVDRAM bottom + 8 DRAM top**), 355.49 W GPU, 0.7× frequency stage. The
**only** change is that the memory dies now carry **active read/write dynamic
power** on top of standby/refresh.

## Active read/write power model

The `-2` runs were **standby-energy only**. Here each memory die additionally
dissipates the dynamic energy of the accesses it serves:

```
P_active(die) = alpha_w * (peak_stack_bandwidth / n_dies) * E_rw(technology)
```

| Parameter | Value | Source |
|---|---|---|
| Memory activity factor `alpha_w` | **0.2421** | Average memory activity factor, Table `tab:activity-factors` (weight-traffic fraction of HBM bandwidth) |
| Peak per-stack bandwidth | **4.4 TB/s** | Fully-stacked (3D) HBM model, not 2.5D |
| DRAM read/write energy | **70 fJ/bit** | given |
| NVDRAM read/write energy | **90 fJ/bit** | given |

With 12 dies this adds **+49.7 mW per DRAM die** (3.333 → 3.383 W) and
**+63.9 mW per NVDRAM die** (1.083 → 1.147 W, +5.9 %). NVDRAM's higher per-bit
energy (90 vs 70 fJ/bit) is partly offset by its much lower standby (no refresh),
so the bottom NVDRAM dies stay the coolest active layers in the stack.

## Result (per stage: GPU peak / lowest DRAM tier)

| Stage | GPU (°C) | step | Lowest-DRAM (°C) |
|---|---|---|---|
| Baseline (3D stacking) | **137.7** | — | 100.3 |
| + HBM base-die removal | 134.4 | −3.3 | 100.4 |
| + HBM stack merging | 117.4 | −17.0 | 93.1 |
| + Top-die thinning | 115.7 | −1.7 | 91.9 |
| + 0.7× GPU frequency | 92.9 | −22.8 | 76.0 |
| + Thermal silicon | **80.8** | −12.2 | 71.1 |

vs. the standby-only waterfall2 final point (GPU 80.4 / DRAM 70.8): the active
read/write adder shifts every stage up by ≈ **0.3 °C**. φ-HBM remains cooler than
the all-DRAM baseline3 at every matched stage (final: 80.8 vs 86.6 °C GPU).

## Figures

- `waterfall.png/.pdf` — GPU peak + lowest DRAM tier at every stage.
- `composite_waterfall.png/.pdf` — φ-HBM vs all-DRAM (baseline3), both stacks in
  one step-line waterfall.
- `layerwise_thermal_si.png/.pdf` — **per-die peak vs stack position** for the
  all-DRAM and φ-HBM stacks, **final (thermal-silicon) stage only**: GPU anchor at
  position 0, then dies 1 (bottom, nearest GPU) → 12 (top); NVDRAM region shaded.
- `heatmaps_thermal_si.png/.pdf`, `surfaces_thermal_si.png/.pdf` — GPU / lowest
  NVDRAM / lowest DRAM layer maps after thermal-silicon optimization.

## Reproduce

```bash
python3 run_waterfall.py          # φ-HBM 4 NV + 8 DRAM, 355.49 W, 0.7× freq, --active
python3 plot_waterfall.py
python3 plot_composite.py         # reads ../NVDRAM_baseline3 + this folder
python3 plot_layerwise.py         # reads only 5_thermal_si of both stacks
python3 plot_heatmaps.py
python3 plot_surfaces.py
```

Same calibration/fidelity caveats as `../NVDRAM_waterfall2`.
