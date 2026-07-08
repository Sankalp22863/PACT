# EXP_DRAM_B — DRAM (HBM) on a 3D compute die, paper BASELINE ("3D thermal penalty")

Reproduces the **un-optimized** baseline of the IEDM paper *"Breaking Thermal
Bottleneck in 3D HBM-on-GPU Integration via System-Technology Co-Optimization"*
(imec, IEDM 2025) — the leftmost **"3D thermal penalty"** bar of the STCO
staircase, **GPU peak ≈ 141.7 °C**.

It is identical to [`EXP_DRAM_R`](../EXP_DRAM_R/) in every way **except the
cross/frame filler material** between and around the four HBM stacks:

| | Cross/frame filler | k (W/m·K) | Meaning |
|---|---|---|---|
| **EXP_DRAM_R** (optimized) | `THERMAL_SI` | 140 | high-k dummy-silicon heat-escape path — the paper's *thermal-silicon optimization* already applied |
| **EXP_DRAM_B** (this, baseline) | `MOLD` | 0.8 | epoxy mold compound; **no escape path**, heat forced up through the low-k DRAM BEOL / hybrid-bond bottleneck |

That single swap is the difference between the optimized point (~130 °C in our
reproduction) and the raw baseline penalty (~141.7 °C in the paper).

## Why this is the correct baseline

The STCO staircase applies optimizations cumulatively; **"thermal silicon
optimization" is one of the later steps** (it drops the paper's peak 99.2 → 87.4
°C). `EXP_DRAM_R` bakes that optimization in from the start (high-k `THERMAL_SI`
fills 31 % of the die as a continuous vertical escape column straight to the
lid), so it can never reach the un-optimized 141.7 °C point. Replacing that
filler with ordinary mold compound removes the escape path and restores the
penalty — everything else (base die present, stacks un-merged, 169 µm top die,
full 414 W GPU, single-side top cooling) is already the baseline configuration.

> **Mold conductivity.** The paper does not publish the exact baseline gap-fill
> value, so `MOLD` uses the industry-standard epoxy-mold-compound value
> **k = 0.8 W/m·K** (`thermalresistivity = 1.25`). Lower it toward 0.5 for a more
> pessimistic penalty, or run `generate.py --optimized` to fall back to
> `THERMAL_SI` (= EXP_DRAM_R).

## Stack (bottom → top)

Same detailed 45-layer, per-paper cross-section as EXP_DRAM_R:

```
        ↑ NoPackage HTC 30,000 W/m²K  (on top of the Lid)        [TOP COOLING]
  Lid (3000 µm, k=400) · TIM (200 µm, k=9.71)
  12 × [ DRAM Die Si (50/169 µm, k=140) · DRAM BEOL (k=0.85) · Hybrid Bond (k=4.8) ]
  HBM Base Si (50 µm, k=140) · HBM Base BEOL (5 µm, k=1.5)
  GPU–HBM uBump (40 µm, k=19.28) · Oxide (1 µm) · BEOL_MXY (1.4 µm)
  GPU FEOL (0.15 µm, k=7.9, 414 W)                               [HEAT SOURCE]
  BSPDN (1.715 µm, k=71)                                         [PACKAGE SIDE — adiabatic]
```

The 2×2 HBM macros are unchanged; only the surrounding cross/frame is now `MOLD`
instead of `THERMAL_SI`.

## Power

Identical to EXP_DRAM_R: GPU 414 W + 4 HBM stacks × 40 W (= 3.33 W/die) = **574 W**.

## Run it

From `../scripts/`:
```bash
python3 run_experiment.py ../EXP_DRAM_B
```
Generator → PACT (steady-state, SuperLU) → figures in `Results/`. This is the
"pure DRAM baseline" reference for [`EXP_NVDRAM_B`](../EXP_NVDRAM_B/).
