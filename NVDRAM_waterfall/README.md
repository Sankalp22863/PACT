# NVDRAM_waterfall — φ-HBM STCO thermal waterfall (rerun)

Rebuilds the **φ-HBM** thermal waterfall of the FROST paper's Fig. 13 by running
PACT once per cumulative STCO optimization and recording the peak compute-die
(GPU) temperature at each step. This is the **first** run of this waterfall.

## What changed vs. the original

The base model is [`../Experiments/EXP_NVDRAM_B`](../Experiments/EXP_NVDRAM_B/) —
the hybrid **4 NVDRAM (bottom) + 8 DRAM (top)** stack on a 414 W GPU compute die,
top-side liquid lid, package side adiabatic.

The **only** deliberate change from the original waterfall: the original applied
**thermal silicon from the very start** (the `EXP_NVDRAM_S` regime, high-k
`THERMAL_SI` inter-stack filler throughout). That is wrong — "thermal-silicon
optimization" is a *later* STCO step. Here the waterfall **starts from the mold
baseline** (`MOLD`, the `EXP_NVDRAM_B` regime) and only switches the
inter-stack filler to thermal silicon at the thermal-silicon step.

The mold conductivity is set to the imec paper's value, **k = 3 W/(m·K)**
("mold compound (3 W/mK) fills the cavities", Chen et al. Sec. II.A) — not the
generic 0.8 W/m·K the original framework assumed. This brings the baseline into
line with the paper (149.5 °C here vs. the paper's 141.7 °C all-DRAM 3D point).

## Waterfall stages (cumulative — each adds one optimization)

| Stage | Optimization added | How it is modeled (`waterfall_generate.py` flag) |
|---|---|---|
| 0 | Baseline (3D stacking) | mold filler, HBM base die present, thick 169 µm top die, 414 W GPU |
| 1 | + HBM base-die removal | `--no-base-die` (drop the 2 HBM-base sublayers) |
| 2 | + HBM stack merging | `--merge-stacks` (inter-stack mold cross → continuous silicon `MERGE_SI`, k=140, per the paper; the 4 stacks stay memory, only the cross material changes) |
| 3 | + Top-die thinning | `--thin-top-die` (top memory die Si 169 µm → 50 µm) |
| 4 | + 0.7× GPU frequency | `--gpu-power 289.8` (= 0.7 × 414 W, **linear** P∝f) |
| 5 | + Thermal-silicon opt. | `--optimized` (inter-stack/frame filler `MOLD` → `THERMAL_SI`, k=140) |

Double-sided cooling is intentionally **left out** for now.

## Result — peak compute-die temperature

Peak GPU temperature = max over the GPU-FEOL layer **under the 2×2 memory stacks**
(the framework's `mask_surround` convention). The mold gap between stacks cooks to
a few-hundred °C as an artifact of the thin, uniform-power GPU layer; it is
excluded because heat physically escapes under the stacks. (By the thermal-silicon
stage the artifact is gone: raw grid max ≈ under-stack peak = 85.5 °C.)

Numbers below are **calibrated** (lid HTC = 56,000, see below) so the all-DRAM
baseline matches the imec 141.7 °C reference.

| Stage | Peak GPU (°C) | Step |
|---|---|---|
| Baseline (3D stacking) | **135.1** | — |
| + Base-die removal | 131.2 | −3.9 |
| + Stack merging | 130.5 | −0.7 |
| + Top-die thinning | 126.5 | −4.0 |
| + 0.7× GPU frequency | 99.1 | −27.4 |
| + Thermal silicon | **85.5** | −13.6 |

**Total: 135.1 → 85.5 °C (−49.6 °C).** Full data in `waterfall_results.csv`;
φ-HBM figure in `waterfall.png`/`.pdf`; **φ-HBM-vs-all-DRAM comparison in
`combined_waterfall.png`/`.pdf`** (`plot_combined.py`).

Compared against the all-DRAM stack in [`../NVDRAM_baseline`](../NVDRAM_baseline/)
(same waterfall, 12 DRAM tiers), φ-HBM runs **~6 °C cooler at every stage**
(baseline 135.1 vs 141.4 °C; final 85.5 vs 92.1 °C) — the NVDRAM benefit.

## Reproduce

```bash
python3 run_waterfall.py     # runs the 6 stages -> waterfall_results.csv, waterfall_log.txt
python3 plot_waterfall.py    # -> waterfall.png / waterfall.pdf
```
Each stage runs in its own `stageN/` subdir (all inputs + grid outputs preserved).
Needs `scipy` + `pandas` (PACT SuperLU steady-state solver).

## Modeling assumptions worth confirming

1. **Stack merging** — now models the paper's mechanism: the inter-stack mold
   cross is replaced by a **continuous silicon region** (`MERGE_SI`, k=140), the
   four stacks stay memory. This strongly cools the inter-stack region (raw grid
   min 128 → 86 °C at that step) but moves the *under-stack peak* only −0.7 °C,
   vs. the paper's −17.6 °C. The gap is a **model limitation, not the mechanism**:
   the paper's merge benefit comes from ~7 % of peripheral GPU power flowing
   *laterally* into the central silicon (paper Fig. 7), but here the GPU is a
   0.15 µm FEOL layer with no thick silicon substrate, so it can't spread heat
   sideways to exploit the silicon bridge. Reproducing the paper's merge magnitude
   would require adding a laterally-conductive GPU substrate.
2. **GPU frequency → power** — modeled **linearly** (0.7× frequency = 0.7× GPU
   power = 289.8 W). A voltage-scaled (super-linear) law would give a larger drop
   at that step; change the `--gpu-power` value to explore.
3. **Absolute calibration (APPLIED)** — the lid HTC is set to an *effective*
   **56,000 W/m²K** (vs the paper's physical 30,000) so the all-DRAM 3D baseline
   lands on the imec 141.7 °C reference (this model: 141.4 °C). This single knob is
   frozen across all stages and both stacks. It was chosen because the raw model's
   absolute is extremely geometry-sensitive (a 10 % GPU-die-area change swings the
   peak ~28 °C — unreliable as an absolute), so a one-parameter calibration (the
   FROST paper's own stated methodology) is the honest way to anchor it. The
   calibration preserves every structural delta and only slightly shrinks the
   frequency step (−30.7 → −27.4, physically correct: better cooling → smaller ΔT
   per watt removed). Set HTC back to 30000 in `experiment.config` for the raw,
   uncalibrated model. Per-step deltas track the paper: base-die removal
   (−3.9 here / −3.7 paper), thermal silicon (−13.6 / −11.8); the outlier is stack
   merging (see assumption 1).
