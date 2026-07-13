# NVDRAM_baseline3 — all-DRAM STCO waterfall **with active read/write** (30 × 22 mm)

The active-access counterpart of [`../NVDRAM_baseline2`](../NVDRAM_baseline2/):
identical paper-faithful geometry (30 × 22 mm, two edge stack columns + central
mold void), identical calibration knobs (substrate 1500 µm, merge-void 5000 µm,
lid HTC 35 000), 414 W GPU, 0.5× frequency stage. The **only** change is the
memory-die power model.

## What's new vs NVDRAM_baseline2 — active read/write power

The `-2` runs were **standby-energy only** (background + temperature-dependent
refresh). Here we add the **active read/write dynamic power** dissipated while the
memory is actually being accessed:

```
P_active(die) = alpha_w * (peak_stack_bandwidth / n_dies) * E_rw(technology)
```

| Parameter | Value | Source |
|---|---|---|
| Memory activity factor `alpha_w` | **0.2421** | Average of the memory activity factors, Table `tab:activity-factors` (weight-traffic fraction of HBM bandwidth) |
| Peak per-stack bandwidth | **4.4 TB/s** | Fully-stacked (3D) HBM model, not 2.5D |
| DRAM read/write energy | **70 fJ/bit** | given |
| NVDRAM read/write energy | **90 fJ/bit** | given (unused here — all-DRAM) |

The per-stack bandwidth is shared uniformly across the memory dies; each die
dissipates its share at its own read/write energy. For the all-DRAM stack this
adds **+49.7 mW/die** (DRAM 3.333 → 3.383 W/die, +1.5 %). The adder is small
because 70 fJ/bit is an *array-level* access energy — consistent with the paper's
own fJ-scale device framing — so background + refresh still dominate memory
self-heating.

## Result (per stage: GPU peak / bottom-most DRAM tier)

| Stage | GPU (°C) | step | Bottom-DRAM (°C) |
|---|---|---|---|
| Baseline (3D stacking) | **143.1** | — | 121.2 |
| + HBM base-die removal | 139.8 | −3.4 | 121.3 |
| + HBM stack merging | 123.0 | −16.8 | 111.4 |
| + Top-die thinning | 121.2 | −1.8 | 110.1 |
| + 0.5× GPU frequency | 99.0 | −22.2 | 91.2 |
| + Thermal silicon | **86.6** | −12.4 | 83.8 |

vs. the standby-only baseline2 final point (GPU 86.3 / DRAM 83.5): the active
read/write adder shifts every stage up by ≈ **0.3 °C**.

## Figures

- `waterfall.png/.pdf` — GPU peak + bottom-most DRAM tier at every stage.
- `heatmaps_thermal_si.png/.pdf`, `surfaces_thermal_si.png/.pdf` — GPU + bottom
  DRAM layer maps after thermal-silicon optimization (no NVDRAM layer here).
- The **layerwise stack profile** (all-DRAM vs φ-HBM, final stage only) lives in
  [`../NVDRAM_waterfall3/layerwise_thermal_si.png`](../NVDRAM_waterfall3/), and
  the **composite waterfall** (both stacks) in
  [`../NVDRAM_waterfall3/composite_waterfall.png`](../NVDRAM_waterfall3/).

## Reproduce

```bash
python3 run_waterfall.py          # all-DRAM, 414 W, 0.5× freq, --active on every stage
python3 plot_waterfall.py
python3 plot_heatmaps.py
python3 plot_surfaces.py
```

Same calibration/fidelity caveats as `../NVDRAM_baseline2` (uniform power,
adiabatic package side, compact solver — absorbed by the frozen calibration knobs).
