# NVDRAM-on-GPU 3D Stack — Thermal Simulation

PACT thermal model of an **N-tier ferroelectric NVDRAM stacked on top of a GPU compute die**,
using an imec-style 3D integration scheme. This experiment is a substantially extended version of
[`../NVDRAM_Experiment/`](../NVDRAM_Experiment/): the memory now sits on a hot logic die, each
device tier is split into FEOL/BEOL sublayers, and lateral **filler** material surrounds the memory
macro.

---

## What changed vs. `NVDRAM_Experiment/`

| Aspect | `NVDRAM_Experiment` | `NVDRAM_Stacked` (this experiment) |
|---|---|---|
| Base die | Memory's own CMOS-under-array | **GPU compute die** (dominant heat source) |
| Tier model | One lumped 1T1C layer (2 µm) | Each tier split into **BEOL + FEOL** sublayers |
| In-plane material | Single material per layer | **Multi-material**: memory macro + **FILLER** frame |
| 3D interface | Monolithic | **Hybrid-bond (HB) microbumps** between GPU and memory |
| Footprint | 8.5 mm die | **17 mm GPU die**, 8.5 mm memory macro centred on it |
| Idle power | (tunable) | **NVDRAM idle = 0** (power-gated, non-volatile); GPU idle ≠ 0 |

No PACT source changes are required — every new layer is just a new material `Label` in the config,
and PACT already supports multiple materials within one floorplan.

---

## Architecture

Heat flows from the GPU **up** through the stack to the heat sink (layer 0 is the package side;
PACT appends the heat spreader + heat sink at the top automatically).

```
                          ↑ Heat Sink  /  Heat Spreader   (added by PACT)
  ┌──────────────────────────────────────────────────────────┐
  │  NVDRAM tier 1   FEOL (1T1C/HZO)   │ filler │  ← layer 11 (nearest sink)
  │                  BEOL (Cu)         │ filler │  ← layer 10
  │      ⋮  (tiers 2 … N, each FEOL over BEOL)  ⋮               │
  │  NVDRAM tier N   FEOL (1T1C/HZO)   │ filler │  ← layer 5
  │                  BEOL (Cu)         │ filler │  ← layer 4
  ├──────────────────────────────────────────────────────────┤
  │  HB microbumps (Cu/underfill)      │ filler │  ← layer 3  (3D interface)
  ├──────────────────────────────────────────────────────────┤
  │  GPU BEOL  (Cu / low-k)                     │  ← layer 2
  │  GPU FEOL/TSV  (active Si — HEAT SOURCE)     │  ← layer 1
  │  GPU substrate/TSV  (bulk Si)               │  ← layer 0  (package side)
  └──────────────────────────────────────────────────────────┘
```

For the default **N = 4** that is **12 device layers** (GPU ×3 + microbumps ×1 + 4 tiers × 2).
Within each tier the **FEOL is above the BEOL** (matching the imec sequence). The memory macro
(8.5 mm) occupies the centre of every memory/HB layer; the surrounding frame is dielectric FILLER.

### FEOL vs BEOL
- **FEOL** (front-end-of-line) = the transistor/device level. This is where power is dissipated
  (GPU compute logic; NVDRAM 1T1C access transistors + HZO ferroelectric caps).
- **BEOL** (back-end-of-line) = the metal interconnect stack (Cu + low-k dielectric). Carries only
  small interconnect power but adds real thermal resistance (low-k k ≈ 12 W/m·K).

### Filler
In imec memory-on-logic, the memory macro covers only part of the larger logic die; the rest of each
memory tier is **dielectric filler** for planarisation/mechanical support. It is modelled as an
insulating material (**k = 1.4 W/m·K**, like an oxide). Thermally this matters: the filler frame
stays cooler than the active macro, and conducts little heat laterally.

---

## Files

### Generator — `generate_stacked.py`
Produces every floorplan, ptrace, and the LCF, parameterised by tier count, GPU power, and die size:

```bash
python3 generate_stacked.py [--n N] [--gpu-tdp W] [--gpu-side M] [--mem-side M]
```
| Arg | Default | Meaning |
|---|---|---|
| `--n` | 4 | Number of NVDRAM tiers (each = BEOL + FEOL) |
| `--gpu-tdp` | 75 | GPU active power in watts |
| `--gpu-side` | 0.017 | GPU die edge length (m) |
| `--mem-side` | 0.0085 | NVDRAM macro edge length (m) |

It prints a stack summary with per-layer thickness and analytic thermal resistance
(`R_th = thickness / (k · area)`).

### Shared 9-block geometry
Every floorplan uses the same 9 blocks so all layers line up. The central 8.5 mm macro keeps the
standard memory layout (`MemArray`, `RowDecoder_L/R`, `SenseAmp_B/T`); four `Frame_*` blocks tile
the surrounding border. Only the `Label` (and power) differ per layer type. With a 40×40 grid on the
17 mm die (0.425 mm/cell), the macro is exactly **20×20 cells**, offset 10 cells (4.25 mm) on each
side — so block edges land on grid lines.

### Materials — `experiment.config`
`thermalresistivity = 1/k`. Seven materials:

| Label | Layer | k (W/m·K) | ρ ((m·K)/W) | Cp (J/m³·K) |
|---|---|---|---|---|
| `GPU_Si` | GPU substrate/TSV | 130 | 0.0077 | 1.75e6 |
| `GPU_FEOL` | GPU active Si | 110 | 0.0091 | 1.70e6 |
| `GPU_BEOL` | GPU Cu/low-k metal | 12 | 0.083 | 2.40e6 |
| `HB_uBump` | Cu-Cu hybrid bond + underfill | 20 | 0.05 | 2.50e6 |
| `FILLER` | dielectric filler/underfill | 1.4 | 0.714 | 1.66e6 |
| `NV_FEOL` | 1T1C poly-Si + HZO devices | 14 | 0.07 | 1.50e6 |
| `NV_BEOL` | memory Cu interconnect | 12 | 0.083 | 2.20e6 |

### Power maps (ptrace) — 3 columns: Idle / Active-Read / Active-Write
- **`gpu_feol_ptrace.csv`** — the dominant source. TDP is spread uniformly by area over the full
  17 mm die (the GPU computes everywhere; the memory just sits over the centre). Active = 75 W,
  **idle ≈ 11 W** (GPU is volatile → leakage even when idle).
- **`nvdram_feol_ptrace.csv`** — per tier, **Idle = 0** (non-volatile, fully power-gated). Read/write
  tuned for ferroelectric physics: write is the peak (HZO polarisation switching + write-verify),
  read is elevated by the destructive-read + write-back of a 1T1C cell. Centre totals ≈ 0.70 W read,
  1.30 W write.
- **`nvdram_beol_ptrace.csv`** — small interconnect power (idle 0, ≈0.15 W read, 0.30 W write).
- **GPU substrate, GPU BEOL, HB microbumps, and all FILLER blocks** — passive (negligible `1e-6 W`).

> **Why GPU idle ≠ 0 but NVDRAM idle = 0:** the NVDRAM is non-volatile, so during idle it can be
> fully power-gated with no data loss and no refresh — its idle power genuinely collapses to ~0. The
> GPU is volatile logic; even idle it leaks (and must retain state), so its idle power stays finite.

### Static `modelParams.config`
Solver (`SPICE_transient` via Xyce by default; comment in `SuperLU` for a faster steady-only run),
grid (40×40), heat-sink package, and one `Solid`-library mapping per material label.

---

## Running

```bash
source ../setenv.sh                     # put Xyce on PATH (once per shell)
python3 generate_stacked.py --n 4 --gpu-tdp 75
python3 ../src/PACT.py stacked_n4_lcf.csv experiment.config modelParams.config \
    --gridSteadyFile stacked_n4.grid.steady
```
The default N=4 transient run takes ~1–2 min with serial Xyce.

### Plotting heatmaps — `plot_heatmap.py`
After a run, render the per-layer steady-state temperature grids:

```bash
python3 plot_heatmap.py                       # all device layers, shared colour scale
python3 plot_heatmap.py --layer 1             # just the GPU FEOL layer
python3 plot_heatmap.py --include-package      # also show heat spreader + sink
python3 plot_heatmap.py --prefix stacked_n4.grid.steady --out figs/stack.png
```
Each panel uses a shared colour scale (so layers are comparable); a dashed box marks the 8.5 mm
NVDRAM macro, with the surrounding region being filler. Layer titles are read from the LCF. Output
defaults to `<prefix>.heatmaps.png`.

### Output files
| File | Contents |
|---|---|
| `stacked_n4.grid.steady.layer{N}` | 40×40 steady-state temperature grid (°C) per layer |
| `stacked_n4.block.transient.csv` | Block temperature vs. time, per layer |
| `stacked_n4.cir.csv` | Full Xyce transient grid trace |
| `stacked_n4.cir.ic` | Final temperature state (initial condition for follow-on runs) |
| `stacked_n4.log` | Xyce solver log |
| `stacked_n4.grid.steady.heatmaps.png` | Per-layer heatmap figure (from `plot_heatmap.py`) |

**Layer numbering:** 0 = GPU substrate (package side) … 11 = top NVDRAM FEOL (nearest sink);
PACT adds layer 12 = heat spreader and layer 13 = heat sink.

---

## Representative result (N=4, 75 W GPU, ambient 45 °C)

Steady-state, active scenario:

| Region | Observation |
|---|---|
| GPU FEOL (layer 1) | Hottest layer (~56 °C peak); the heat source |
| Gradient up the stack | Mean temp falls monotonically GPU → top tier → sink |
| Macro vs. filler | The central memory macro runs hotter than the insulating FILLER frame, and the gap **grows up the stack** (≈ +1.6 K at the GPU to ≈ +3.6 K at the top tier) — the GPU's heat is funneled up through the memory macro while the filler stays cool |

This captures the core thermal story of stacking memory on a hot logic die: the buried memory tiers
are heated from below by the GPU, the FEOL/BEOL/microbump interfaces each add resistance, and the
filler creates a strong in-plane hot-macro / cool-frame contrast.

> **Note:** All power values are representative engineering estimates (chosen for realistic relative
> and order-of-magnitude behaviour), **not** circuit-extracted numbers. Tune them, the materials, or
> the geometry via `generate_stacked.py` / `experiment.config` for a specific design.
