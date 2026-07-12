# EXP_NVDRAM_S — N stacked NVDRAM below DRAM on a 3D-stacked compute die

A "future 3D integration" where **both** memory technologies are stacked directly
on top of a GPU compute die and cooled by a top-side liquid lid:

- the **non-volatile NVDRAM** tiers sit at the **bottom** of the memory stack
  (nearest the GPU), and
- the normal **(HBM) DRAM** tiers sit on **top** (nearest the lid).

The package side (bottom of the GPU) is adiabatic, so heat is forced **up**
through the whole memory stack to the lid.

## Stack (bottom → top) — detailed per-paper cross-section

Mirrors EXP_DRAM_R's detailed stack (45 device layers), but the bottom memory
tiers are NVDRAM (replacing that many DRAM tiers). GPU/interface/TIM/Lid are
full-die; memory sublayers use the paper's Fig. 3(a) corner layout (four 11×11 mm
stacks, two per short edge of the 30×22 mm die, 8 mm central thermal-Si column).

```
        ↑ NoPackage HTC 30,000 W/m²K  (on top of the Lid)        [TOP COOLING]
  Lid (3000 µm, k=400) · TIM (200 µm, k=9.71)
  ┌ 8 × DRAM tier  [ DRAM Die Si (50/169 µm,k=140) · DRAM BEOL (k=0.85) · Hybrid Bond (k=4.8) ]
  └ 4 × NVDRAM tier[ NVDRAM Die Si (50 µm,k=140)   · NV BEOL (k=0.85)   · Hybrid Bond (k=4.8) ]
  HBM Base Si (50 µm,k=140) · HBM Base BEOL (5 µm,k=1.5)
  GPU–HBM uBump (40 µm, k=19.28)
  Oxide (1 µm) · BEOL_MXY (1.4 µm)
  GPU FEOL (0.15 µm, k=7.9, 414 W)                               [HEAT SOURCE]
  BSPDN (1.715 µm, k=71)                                         [PACKAGE SIDE — adiabatic]
```

Default composition: **4 NVDRAM tiers + 8 DRAM tiers** (= 12-Hi). Four memory
stacks flush in the die corners with a central thermal-silicon (k=140) escape column, so a
cross-section shows `memory | thermal-silicon | memory`.

## Materials (`experiment.config`)

Identical detailed material set to EXP_DRAM_R (cross-plane k from the paper
table) **plus the NVDRAM die**:

| Label | k (W/m·K) | Role |
|---|---|---|
| NV_DIE_SI | **140** | NVDRAM die Si — *equal to DRAM* (canonical; see note) |
| NV_DIE_BEOL | 0.85 | NVDRAM die BEOL |
| DRAM_SI | 140 | DRAM die Si (inner 50 µm / top 169 µm) |
| HYBRID_BOND | 4.8 | Cu/oxide hybrid bond between dies |
| THERMAL_SI | 140 | thermal-silicon escape paths |

> **NVDRAM conductivity.** `NV_DIE_SI` is set **equal to the DRAM die, k = 140**
> (`thermalresistivity = 0.007143`) — the canonical choice, so the technologies
> differ only in *power*. Lower it (e.g. `0.07` → k≈14) to model a low-thermal-
> conductivity ferroelectric NVDRAM, in which case the bottom tiers run *hotter*
> than DRAM despite the lower power.

## Power model — standby only (leakage + refresh)

Memory power is modelled as **background/standby power**, capturing the key
asymmetry between the two technologies:

```
P_DRAM die   = leakage + refresh
P_NVDRAM die = nv_leakage_factor × (DRAM leakage)  +  0 refresh   (non-volatile)
            ⇒  P_NVDRAM  <  P_DRAM
```

NVDRAM is non-volatile, so its **refresh power is zero** and its **leakage is lower**.
Dynamic read/write activity is not modelled (the GPU is the dominant heat source).

**Replacement semantics.** The memory is a fixed `(nv_tiers + dram_tiers)`-Hi stack
carrying the paper's `dram_per_stack` budget, so each DRAM die = `dram_per_stack /
total_tiers` — the **same per-die power as the pure-DRAM stack**. The bottom NVDRAM
dies *replace* that many DRAM dies (removing their heat), each dissipating less.

With the defaults (`--dram-per-stack 40`, `--dram-refresh-frac 0.35`,
`--nv-leakage-factor 0.5`) and a 12-Hi stack (4 NVDRAM + 8 DRAM):

| Per die | Value |
|---|---|
| DRAM (40/12) | 3.33 W  (leakage 2.17 + refresh 1.17) |
| NVDRAM | **1.08 W**  (0.5 × DRAM leakage, no refresh) |

Totals: GPU 414 W + DRAM 106.7 W + NVDRAM 17.3 W = memory **124 W** — *less* than an
all-DRAM 12-Hi stack (160 W), because the bottom 4 tiers were swapped to lower-power
NVDRAM. Tunable generator args: `--nv-tiers`, `--dram-tiers`, `--gpu-power`,
`--dram-per-stack`, `--dram-refresh-frac`, `--nv-leakage-factor`.

## Run it

From `../scripts/` (run **EXP_DRAM_R first** — the comparison plot needs it):
```bash
python3 run_experiment.py ../EXP_DRAM_R
python3 run_experiment.py ../EXP_NVDRAM_S
```
Produces `generate.py` inputs → PACT (steady-state, SuperLU) → figures in
`Results/`, including `peak_per_tier_hybrid_vs_dram.png`.

## Key result

Both this experiment and EXP_DRAM_R now use the **same detailed stack**, so the
comparison is apples-to-apples. With NVDRAM at DRAM-equal conductivity (canonical,
k=140) and the replacement-heat model, swapping the bottom 4 DRAM tiers for
lower-power NVDRAM **removes heat** (memory 124 W vs an all-DRAM 160 W), so the
hybrid runs **cooler than pure DRAM at every tier**: base tier **115.2 °C vs
122.7 °C (−7.5 °C)**. `peak_per_tier_hybrid_vs_dram.png` shows the hybrid profile
(NVDRAM bottom → DRAM continuing on top) sitting below the pure-DRAM reference.

Lower `NV_DIE_SI` to the physical **k ≈ 14** (`thermalresistivity = 0.07`) and the
result flips: the low NVDRAM conductivity dominates and the bottom tiers run hotter
than pure DRAM despite the lower power — the "thermal bottleneck" regime.
