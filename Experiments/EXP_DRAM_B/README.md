# EXP_DRAM_B — DRAM (HBM) on a 3D compute die, paper BASELINE ("3D thermal penalty")

Reproduces the **un-optimized** baseline of the IEDM paper *"Breaking Thermal
Bottleneck in 3D HBM-on-GPU Integration via System-Technology Co-Optimization"*
(imec, IEDM 2025) — the leftmost **"3D thermal penalty"** point of the STCO
staircase, **GPU peak = 141.7 °C** (this reproduction: **141.7 °C**).

It is identical to [`EXP_DRAM_R`](../EXP_DRAM_R/) except for the two ways the
paper's baseline differs from the merged/optimized case:

| | EXP_DRAM_R (merged/optimized) | EXP_DRAM_B (this, baseline) |
|---|---|---|
| Inter-stack gap | none — the two stacks on each short edge are merged | **mold gap** between the two stacks on each edge (paper: ~0.1 mm of k=3 mold; the paper's *HBM stack merging* step removes it, worth 17.6 °C) |
| GPU power map | uniform 414 W (idealized reference) | **structured non-uniform map**: compute-cluster lattice with correlated random utilisation, asymmetric activity tilt, powered central L2/NoC column, IO edge ring |

The **central 8 mm column stays `THERMAL_SI`** in both — the paper's baseline
already has it ("thermal silicon fills the central void").

## Modelling notes

- **Equivalent-k mold gap.** The 0.5 mm grid cannot resolve the paper's ~0.1 mm
  gap, so the generator draws it 1.0 mm wide with k scaled ×10
  (**k_eff = 30 W/m·K**, see `[MOLD]` in `experiment.config`) to preserve the
  same *lateral* thermal resistance across the gap.
- **Structured power map** (`scripts/stack_common.py`). The paper applies
  commercial 0.5 mm-resolution power maps, which are not public. The stand-in
  is a compute-cluster lattice with spatially-correlated seeded-random
  utilisation, an asymmetric large-scale activity tilt, a powered (×0.7)
  central L2/NoC column, and a low-power IO edge ring, at 0.5 mm tiles. The
  tile **peak-to-average ratio = 2.19** (a realistic hotspot intensity) is the
  single knob, calibrated so this baseline reproduces the paper's 141.7 °C GPU
  peak; tune with `--map-p2a`, change the pattern with `--map-seed`, or use
  `--power-map uniform` for a flat 414 W map.
- **Package surround.** All layers are embedded in a 6 mm `PKG_MOLD` ring
  (k = 3) with the copper lid extended over the whole 42×34 mm package, so heat
  spreads laterally beyond the die as in the paper's 65×65 mm package. The
  laminate (bottom) face remains adiabatic — PACT exposes one convective
  boundary, and the paper's air-cooled laminate (200 W/m²K) carries only a
  small share of the 574 W.
- **Anisotropy.** The paper's two anisotropic layers use their true in-plane
  conductivities via the optional `inplanethermalresistivity` property
  (uBump: in-plane k = 0.59, cross 19.28; BSPDN: in-plane 83, cross 71).
- `--merged --power-map uniform` together reproduce EXP_DRAM_R.

## Stack (bottom → top)

Same detailed 45-layer, per-paper cross-section as EXP_DRAM_R:

```
        ↑ NoPackage HTC 30,000 W/m²K  (on top of the Lid)        [TOP COOLING]
  Lid (3000 µm, k=400) · TIM (200 µm, k=9.71)
  12 × [ DRAM Die Si (50/169 µm, k=140) · DRAM BEOL (k=0.85) · Hybrid Bond (k=4.8) ]
  HBM Base Si (50 µm, k=140) · HBM Base BEOL (5 µm, k=1.5)
  GPU–HBM uBump (40 µm, k=19.28) · Oxide (1 µm) · BEOL_MXY (1.4 µm)
  GPU FEOL (0.15 µm, k=7.9, 414 W hotspot map)                   [HEAT SOURCE]
  BSPDN (1.715 µm, k=71)                                         [PACKAGE SIDE — adiabatic]
```

Lateral layout per paper Fig. 3(a): 30×22 mm GPU, four 11 mm-wide HBM stacks in
the corners (two per short edge, 1 mm mold gap between them), 8 mm central
thermal-silicon column.

## Power

GPU 414 W (hotspot map, p2a 1.68) + 4 HBM stacks × 40 W (= 3.33 W/die) = **574 W**.

## Run it

From `../scripts/`:
```bash
python3 run_experiment.py ../EXP_DRAM_B
```
Generator → PACT (steady-state, SuperLU) → figures in `Results/`. This is the
"pure DRAM baseline" reference for [`EXP_NVDRAM_B`](../EXP_NVDRAM_B/).
