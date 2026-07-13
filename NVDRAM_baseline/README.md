# NVDRAM_baseline — all-DRAM STCO thermal waterfall (comparison stack)

The **all-DRAM** counterpart of [`../NVDRAM_waterfall`](../NVDRAM_waterfall/):
the identical HBM-on-GPU stack and the identical STCO waterfall, but with **12
DRAM tiers and zero NVDRAM** (`--nv-tiers 0 --dram-tiers 12`). This is the
homogeneous-DRAM reference the φ-HBM (hybrid) run is compared against — the
analogue of the all-DRAM curve in the paper's Fig. 13 / the imec STCO staircase.

Everything else is shared with `NVDRAM_waterfall` (same generator, materials —
including the paper's mold k = 3 W/m·K and the continuous-silicon `MERGE_SI`
merge — top-side liquid lid, adiabatic package side, under-stack peak metric).

## Result — peak compute-die temperature

Numbers are **calibrated** (lid HTC = 56,000) so this baseline matches the imec
141.7 °C reference (this model: 141.4 °C).

| Stage | Peak GPU (°C) | Step |
|---|---|---|
| Baseline (3D stacking) | **141.4** | — |
| + Base-die removal | 137.5 | −3.9 |
| + Stack merging | 136.7 | −0.8 |
| + Top-die thinning | 132.6 | −4.1 |
| + 0.7× GPU frequency | 105.2 | −27.4 |
| + Thermal silicon | **92.1** | −13.1 |

**Total: 141.4 → 92.1 °C (−49.3 °C).**

## φ-HBM vs. all-DRAM

At every matched stage the hybrid φ-HBM stack (`../NVDRAM_waterfall`) runs
**~6 °C cooler** than this all-DRAM stack — the NVDRAM benefit (lower-power,
refresh-free bottom dies). The combined figure is
`../NVDRAM_waterfall/combined_waterfall.png`.

| Stage | All-DRAM (this) | φ-HBM | Δ |
|---|---|---|---|
| Baseline | 141.4 | 135.1 | −6.3 |
| + Base-die removal | 137.5 | 131.2 | −6.3 |
| + Stack merging | 136.7 | 130.5 | −6.2 |
| + Top-die thinning | 132.6 | 126.5 | −6.1 |
| + 0.7× GPU frequency | 105.2 | 99.1 | −6.1 |
| + Thermal silicon | 92.1 | 85.5 | −6.6 |

## Reproduce

```bash
python3 run_waterfall.py --nv-tiers 0 --dram-tiers 12 --label "All-DRAM: 12 DRAM"
python3 plot_waterfall.py
```

## Calibration

The lid HTC is an *effective* **56,000 W/m²K** (vs the paper's physical 30,000),
chosen so this all-DRAM 3D baseline matches the imec-reported 141.7 °C
(→ 141.4 °C here). Rationale: the raw model (HTC = 30,000) gives 156.8 °C, and its
absolute is extremely geometry-sensitive (a 10 % GPU-die-area change swings the
peak ~28 °C), so it is not trustworthy as an absolute — a single frozen
calibration knob (the FROST paper's own stated methodology) is the honest anchor.
The knob is held fixed across all stages and both stacks, so the per-step STCO
deltas and the ~6 °C φ-HBM-vs-DRAM offset are preserved. The gap the calibration
absorbs is the compact-model-vs-Ansys-Icepak difference plus this model's thin
uniform-power GPU layer (no lateral spreading), adiabatic package side, and 2×2
four-stack geometry. Set HTC back to 30,000 in `experiment.config` for the raw
model.
