# NVDRAM_waterfall2 — φ-HBM STCO waterfall, paper-faithful layout (30 × 22 mm)

The **φ-HBM (4 NVDRAM + 8 DRAM)** counterpart of the faithful all-DRAM
[`../NVDRAM_baseline2`](../NVDRAM_baseline2/): identical paper-faithful geometry
(reference Fig. 3 — two stack columns on the short edges + central mold void),
identical calibration knobs, run on the φ-HBM stack.

## What's different from NVDRAM_baseline2

| | NVDRAM_baseline2 | **this (φ-HBM)** |
|---|---|---|
| Memory stack | 12 DRAM | **4 NVDRAM (bottom) + 8 DRAM (top)** |
| GPU power | 414 W | **355.49 W** (power density 5.38625e-7 W/µm² × 30 × 22 mm) |
| Frequency stage | 0.5× → 300 W | **0.7× → 297 W** (paper Fig. 8: 346 W at 0.7× for a 414 W GPU, scaled to 355.49 W) |
| Calibration (substrate 1500 µm, void 5000 µm, HTC 35 000) | same | same (apples-to-apples) |

## Result

Peak GPU = global max on the GPU-FEOL layer; lowest-DRAM = max under the stacks
on the bottom-most DRAM tier (added to the waterfall at every stage).

| Stage | GPU (°C) | step | Lowest-DRAM (°C) |
|---|---|---|---|
| Baseline (3D stacking) | **137.4** | — | 99.9 |
| + HBM base-die removal | 134.1 | −3.3 | 100.0 |
| + HBM stack merging | 117.0 | −17.1 | 92.7 |
| + Top-die thinning | 115.3 | −1.7 | 91.5 |
| + 0.7× GPU frequency | 92.6 | −22.8 | 75.6 |
| + Thermal silicon | **80.4** | −12.2 | 70.8 |

Merge = **−17.1 °C** (reference −17.6), as in the calibrated all-DRAM run.

The φ-HBM baseline (137.4 °C) is ~5.4 °C below the all-DRAM baseline (142.8 °C) —
though note this run's GPU is 355.49 W vs. baseline2's 414 W, so the difference
reflects both the lower GPU power and the lower-power/refresh-free NVDRAM bottom
tiers, not the memory swap alone.

## Figures

- `waterfall.png/.pdf` — GPU peak (purple) + lowest DRAM tier (blue) at every stage.
- `heatmaps_thermal_si.png/.pdf` — **GPU (80.4 °C) / lowest NVDRAM (77.5 °C) /
  lowest DRAM (70.8 °C)** layer temperature maps after thermal-silicon optimization
  (layers 2 / 8 / 20).

## Reproduce

```bash
python3 run_waterfall.py                 # defaults: 4 NVDRAM + 8 DRAM, 355.49 W
python3 plot_waterfall.py                # -> waterfall.png/.pdf
python3 plot_heatmaps.py                 # -> heatmaps_thermal_si.png/.pdf
```

Same calibration and fidelity caveats as `../NVDRAM_baseline2` (uniform power,
adiabatic package side, compact solver — absorbed by the frozen calibration knobs).
