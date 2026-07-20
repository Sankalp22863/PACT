# NVDRAM_waterfall3 — φ-HBM STCO waterfall **with active read/write** (30 × 22 mm)

The active-access counterpart of [`../NVDRAM_waterfall2`](../NVDRAM_waterfall2/):
identical paper-faithful geometry and calibration, φ-HBM stack
(**4 NVDRAM bottom + 8 DRAM top**), 414 W GPU, 0.8× frequency stage (368 W, paper Fig. 8). The
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
| + 0.8× GPU frequency | 106.7 | −9.0 | 85.6 |
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
- `nvdram_replacement_tradeoff.png/.pdf` — peak DRAM temperature (measured) and
  decode throughput (modelled) vs the number of FeRAM dies, N = 0…12.
- `nvdram_reliability.png/.pdf` — see below.

## Thermal reliability (`plot_reliability.py`)

Three thermally-activated mechanisms evaluated on the **measured** per-die
temperatures of `replace_sweep/nv4` (the XBM N = 4 operating point). The thermal
model is untouched — temperatures are inputs, reliability is modelled on top.

| Mechanism | Basis | Result at the hottest FeRAM die (94.7 °C) |
|---|---|---|
| FeRAM write endurance | `T_eff` (wear-levelled ensemble) | 6.5 × 10¹¹ cycles → **~700× margin** even at KV-rate writes |
| **FeRAM retention** | peak T (hottest cell fails first) | **4.3 yr** vs a 10-yr life — all four FeRAM dies fall short (4.3–7.9 yr) |
| DRAM retention | peak T | 63 ms at the 85.2 °C DRAM peak; derives the 85/90 °C limits used elsewhere |

Two corrections to earlier framing came out of this:

1. **Endurance does not justify pinning the KV-cache to DRAM.** It clears KV-rate
   writes by ~700×. The binding reason is **write energy** (90 fJ/bit plus the
   destructive-read write-back). `plot_replacement_tradeoff.py` has been updated.
2. **FeRAM is not thermally unconstrained.** It needs no refresh, but retention is
   finite and Arrhenius-activated; at the bottom-die temperature it is 4.3 yr.
   Practically this means the weights must be rewritten every few years — trivially
   satisfied by normal model reloads, but it must be stated rather than assumed away.

Key parameters (all module constants, all reported in the figure footnote):
N₀ = 10¹² cycles @ 85 °C, Ea 0.3–0.8 eV (fatigue) and 0.8–1.3 eV (retention),
0.45 Gb/mm² × 506 mm² = 28.5 GB/die, 4.4 TB/s/stack × α_w = 0.2421.
`F_WRITE` is new and lives **only** in this model — the power model correctly
charges 90 fJ per access regardless of direction.

```bash
python3 plot_reliability.py --verify   # 13 self-checks
python3 plot_reliability.py            # writes nvdram_reliability.png/.pdf
```

## Reproduce

```bash
python3 run_waterfall.py          # φ-HBM 4 NV + 8 DRAM, 414 W full, 0.8× freq (368 W), --active
python3 plot_waterfall.py
python3 plot_composite.py         # reads ../NVDRAM_baseline3 + this folder
python3 plot_layerwise.py         # reads only 5_thermal_si of both stacks
python3 plot_heatmaps.py
python3 plot_surfaces.py
```

Same calibration/fidelity caveats as `../NVDRAM_waterfall2`.
