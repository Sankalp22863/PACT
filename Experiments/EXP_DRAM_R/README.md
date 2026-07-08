# EXP_DRAM_R — DRAM (HBM) on a 3D-stacked compute die (paper reproduction)

Reproduces the thermal model from the IEDM paper *"Breaking Thermal Bottleneck
in 3D HBM-on-GPU Integration via System-Technology Co-Optimization"*: four 12-Hi
HBM (DRAM) stacks integrated **directly on top** of a GPU compute die, cooled by
a top-side liquid cold plate (the "lid"). The package/laminate side (bottom of
the GPU) is adiabatic, so heat is forced **up** through the HBM stacks to the lid
— the paper's "thermal bottleneck" regime.

## Stack (bottom → top) — detailed per-paper cross-section

Every physical sublayer from the paper's cross-section table is modelled (45
device layers). GPU/interface/TIM/Lid are full-die; the HBM-stack sublayers use
the 2×2 macro + thermal-silicon cross/frame layout.

```
        ↑ NoPackage HTC 30,000 W/m²K  (on top of the Lid)        [TOP COOLING]
  Lid              3000 µm, k=400
  TIM               200 µm, k=9.71
  ┌ 12 × [ DRAM Die Si (50 µm, top 169 µm, k=140)
  │        DRAM Die BEOL (3 µm, k=0.85)
  │        Hybrid Bonding (2 µm, k=4.8) ]                         (2×2 + thermal-Si)
  └ HBM Base Si (50 µm, k=140) + HBM Base BEOL (5 µm, k=1.5)
  GPU–HBM uBump     40 µm, k=19.28
  Oxide              1 µm, k=1.5
  BEOL_MXY         1.4 µm, k=1.5
  GPU FEOL        0.15 µm, k=7.9, 414 W                           [HEAT SOURCE]
  BSPDN          1.715 µm, k=71                                   [PACKAGE SIDE — adiabatic]
```

The orientation is 3D-on-GPU (GPU front-side up: uBump on top of the GPU, BSPDN
on the adiabatic bottom). The 2.5D package below the diagram (interposer, Cu
pillar bump, laminate) is **not** modelled — in 3D-on-GPU it is off the heat path.

## Lateral layout — memory on both sides, thermal silicon in the middle

Four HBM stacks in a **2×2** array on the 24 mm GPU die; the cross/frame between
them is high-k **thermal silicon** (k=140), so a vertical cross-section shows
`HBM | thermal-silicon | HBM`. Inside each stack, the low-k **DRAM Die BEOL
(0.85)** and **Hybrid Bonding (4.8)** are the dominant vertical resistances (the
bottleneck).

## Materials (`experiment.config`, cross-plane k from the paper table)

| Label | k (W/m·K) | Label | k (W/m·K) |
|---|---|---|---|
| BSPDN | 71 | HYBRID_BOND | 4.8 |
| GPU_FEOL | 7.9 | DRAM_BEOL | 0.85 |
| BEOL_MXY | 1.5 | DRAM_SI | 140 |
| OXIDE | 1.5 | THERMAL_SI | 140 |
| GPU_HBM_UBUMP | 19.28 | TIM | 9.71 |
| HBM_BASE_SI | 140 | LID | 400 |
| HBM_BASE_BEOL | 1.5 | | |

PACT is isotropic per material, so anisotropic table entries use the **cross-plane**
(vertical) value: uBump 0.59/**19.28**, BSPDN 83/**71**.

## Power (generator defaults, `generate.py`)

GPU 414 W (active heat source) + 4 HBM stacks × 40 W = **574 W total**; the 40 W
per 12-Hi stack is split per die (40/12 = 3.33 W) on the DRAM Die Si macros. Base
die passive. Args: `--gpu-power`, `--hbm-per-stack`, `--tiers`, `--gpu-side`,
`--hbm-side`, `--no-base-die` (the paper's "base-die removal" study).

## Run it

From `../scripts/`: `python3 run_experiment.py ../EXP_DRAM_R` → `generate.py` →
PACT (steady-state, SuperLU, ~1 min for 45 layers) → figures in `Results/`.

## Result

GPU FEOL is the hottest layer at **~131 °C** (vs the paper's ~120 °C GPU peak); the
bottom DRAM die sits at ~123 °C, and temperature falls monotonically up the stack
to the lid. The explicit TIM + low-k DRAM BEOL / hybrid-bond layers add the series
resistance that the earlier coarse model (~114 °C) lacked. `Results/cross_section.png`
shows the full layer stack; `EXP_NVDRAM_S/Results/peak_per_tier_hybrid_vs_dram.png`
uses these DRAM tiers as the "pure DRAM" reference.
