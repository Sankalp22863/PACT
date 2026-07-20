"""
Thermal reliability of the XBM stack — endurance & retention vs temperature
===========================================================================
Applies three thermally-activated reliability models to the MEASURED per-die
temperatures of the 12-high memory stack (PACT grids in replace_sweep/nvN and
5_thermal_si). Nothing here changes the thermal model — temperatures are inputs.

  (a) FeRAM write endurance   — Arrhenius cycles-to-failure, N_f(T).
      Presented INVERTED as the endurance-limited sustainable write bandwidth
      B_max(T, L) = N_f(T)*bits_per_die / (L*year), because at a 1e12-cycle
      rating the raw lifetime is ~10^3-10^4 years and a bandwidth headroom is
      the honest way to show a margin that large.
  (b) FeRAM retention / imprint — Arrhenius, anchored at 10 yr @ 85 C.
      THIS is the mechanism that actually binds: at the bottom FeRAM die
      (94.7 C at N=4) retention falls to ~3-5 yr, short of a 10-yr life.
  (c) DRAM retention — halves per ~10 C. Puts physics under the 85/90 C limits
      that the rest of the repo hardcodes as bare scalars.

IMPORTANT — this script does NOT touch the power/thermal model. The generators
charge 90 fJ per access regardless of direction (read == write), which is correct
for POWER. Endurance is consumed by writes only, so F_WRITE below is a new
parameter that exists ONLY in this reliability model. stack_common.py and
waterfall_generate.py are deliberately left alone; adding F_WRITE there would
change simulated temperatures and invalidate every existing run.

Output: nvdram_reliability.png / .pdf
Self-check:  python3 plot_reliability.py --verify
"""

import argparse
import csv
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "replace_sweep")
PREFIX = "hybrid.grid.steady"

# ── geometry (copied from Experiments/scripts/plot_nvdram_freq_comparison.py:49;
#    that stack_mask() is byte-identical in 5 files and lives in another tree with
#    no sys.path link, so a 6th copy beats refactoring 5 working figure scripts.
#    Upgrade: the central void is derived per-run instead of hardcoded 0.005) ────
GPU_X, GPU_Y = 0.030, 0.022
ROWS, COLS = 44, 60
EDGE_INSERT = 0.001
DEFAULT_VOID = 0.005

INK, MUTE = "#2b2b2b", "#8a8a8a"
C_FERAM, C_DRAM, C_TEAL = "#e08214", "#c0392b", "#1f6f8b"
HL_FC, HL_EC = "#fff3cd", "#e0a800"

# ── physical constants / model parameters ────────────────────────────────────
KB_EV_K = 8.617333262e-5          # Boltzmann constant, eV/K

# FeRAM write endurance (vendor-class HfO2 rating; see caveats in the footnote)
N0_CYCLES, T0_C = 1.0e12, 85.0
EA_END_EV = 0.50
EA_END_SWEEP = (0.30, 0.50, 0.80)  # literature spread for ferroelectric fatigue

# FeRAM retention / imprint (non-volatile data lifetime)
RET_YR_AT_T0 = 10.0
EA_RET_EV = 1.00
EA_RET_SWEEP = (0.80, 1.00, 1.30)

# DRAM retention: JEDEC-style, halves per ~10 C (64 ms at 85 C)
TREF_MS_AT_85, HALVING_C = 64.0, 10.0

# capacity — NVDRAM paper: 0.45 Gb/mm^2 at 48 nm pitch, CMOS under array
DENSITY_GB_PER_MM2 = 0.45
MEM_AREA_MM2 = 506.0               # 4 x (11.5 x 11) mm; asserted against the floorplan
BITS_PER_DIE = DENSITY_GB_PER_MM2 * 1e9 * MEM_AREA_MM2      # 2.277e11 bits = 28.5 GB
WEAR_LEVEL_EFF = 1.0               # fraction of cells the wear-leveller spreads over

# traffic — 4.4 TB/s is the value that GENERATED these temperatures (see nvN/gen.log);
# stack_common.STACK_BW_TBS = 4.9 is the un-derated HBM3E x4 peak, not the operating point.
STACK_BW_TBS, ALPHA_W, TOTAL_TIERS = 4.4, 0.2421, 12
F_WRITE_WEIGHTS = 1.0e-4           # weights are rewritten only on model reload
F_WRITE_KV = 1.0                   # KV-cache: every accessed line is rewritten
SECONDS_PER_YEAR = 3.15576e7
LIFE_TARGET_YEARS = (5.0, 10.0)


# ── floorplan / grid helpers ─────────────────────────────────────────────────
def _read_flp(path):
    """(name, x, y, length, width, label) per floorplan row. Clone of
    plot_nvdram_freq_comparison.py:88."""
    out = []
    for line in open(path).read().splitlines()[1:]:
        c = [t.strip() for t in line.split(",")]
        if len(c) >= 7:
            out.append((c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), c[6]))
    return out


def void_width(grid_dir):
    """Central merge-void width from the run's OWN floorplan, so the mask stays
    correct for any --void-um (the 0.005 hardcode in the sibling scripts is only
    valid for --void-um 5000)."""
    flp = os.path.join(grid_dir, "nv_tier_flp.csv")
    if not os.path.exists(flp):
        flp = os.path.join(grid_dir, "dram_tier_flp.csv")
    for name, _x, _y, length, _w, _lab in _read_flp(flp):
        if name.lower().startswith("void"):
            return length
    return DEFAULT_VOID


def stack_mask(central_void=DEFAULT_VOID):
    """Boolean (ROWS, COLS) mask of the two memory-stack columns."""
    gl = GPU_X / COLS
    cw = (GPU_X - 2 * EDGE_INSERT - central_void) / 2.0
    xs = [(EDGE_INSERT, EDGE_INSERT + cw),
          (EDGE_INSERT + cw + central_void, EDGE_INSERT + cw + central_void + cw)]
    m = np.zeros((ROWS, COLS), dtype=bool)
    for x0, x1 in xs:
        m[:, round(x0 / gl):round(x1 / gl)] = True
    return m


def die_temps(grid_dir):
    """[(pos, tech, peak_C, mean_C, cells_C), ...] bottom -> top for the memory dies.

    Extends profile() (plot_nvdram_freq_comparison.py:63) by keeping the masked
    cell array, so peak / masked-mean / T_eff are all available without re-reading
    the grids (T_eff depends on Ea, so it cannot be precomputed here)."""
    mask = stack_mask(void_width(grid_dir))
    mem = []
    with open(os.path.join(grid_dir, "hybrid_lcf.csv")) as fh:
        next(fh)
        for line in fh:
            idx, flp = [c.strip() for c in line.split(",")[:2]]
            if flp == "nv_tier_flp.csv":
                mem.append((int(idx), "FeRAM"))
            elif flp == "dram_tier_flp.csv":
                mem.append((int(idx), "DRAM"))
    mem.sort()
    out = []
    for pos, (lyr, tech) in enumerate(mem):
        g = (np.loadtxt(os.path.join(grid_dir, f"{PREFIX}.layer{lyr}")) - 273.15).reshape(ROWS, COLS)
        cells = g[mask]
        out.append((pos + 1, tech, float(cells.max()), float(cells.mean()), cells))
    return out


def t_eff_C(cells_C, ea_ev):
    """Thermally-weighted effective temperature: the ensemble aggregate a
    wear-levelled die actually fails at.

    Under wear-levelling every cell takes the same write count, and each
    accumulates damage at rate 1/N_f(T_i) ~ exp(-Ea/kT_i). The die's effective
    endurance is therefore the HARMONIC mean of the per-cell lifetimes, i.e.
        exp(-Ea/k*T_eff) = mean( exp(-Ea/k*T_i) )
    Because that average is dominated by the hot tail, mean <= T_eff <= peak."""
    tk = np.asarray(cells_C, dtype=float) + 273.15
    a = ea_ev / KB_EV_K
    return -a / math.log(float(np.mean(np.exp(-a / tk)))) - 273.15


# ── the three reliability models ─────────────────────────────────────────────
def _arrhenius(value_at_t0, T_C, t0_C, ea_ev):
    """Thermally-activated quantity: hotter -> smaller."""
    return value_at_t0 * math.exp((ea_ev / KB_EV_K) *
                                  (1.0 / (T_C + 273.15) - 1.0 / (t0_C + 273.15)))


def n_f(T_C, n0=N0_CYCLES, t0_C=T0_C, ea_ev=EA_END_EV):
    """FeRAM write cycles-to-failure at temperature."""
    return _arrhenius(n0, T_C, t0_C, ea_ev)


def feram_retention_yr(T_C, ret_yr=RET_YR_AT_T0, t0_C=T0_C, ea_ev=EA_RET_EV):
    """FeRAM non-volatile retention (years) at temperature."""
    return _arrhenius(ret_yr, T_C, t0_C, ea_ev)


def dram_retention_ms(T_C, ref_ms=TREF_MS_AT_85, t0_C=T0_C, halving_C=HALVING_C):
    """DRAM cell retention (ms): halves per `halving_C` degrees."""
    return ref_ms * 2.0 ** (-(T_C - t0_C) / halving_C)


def write_bps_per_die(f_write, alpha=ALPHA_W, bw_tbs=STACK_BW_TBS, tiers=TOTAL_TIERS):
    """Write traffic per die (bit/s). F_WRITE splits the access stream into
    writes; it exists only here, never in the power model."""
    return f_write * alpha * bw_tbs * 1e12 * 8.0 / tiers


def life_years(T_C, write_bps, *, n0=N0_CYCLES, t0_C=T0_C, ea_ev=EA_END_EV,
               bits=BITS_PER_DIE, wl=WEAR_LEVEL_EFF):
    """Endurance-limited life (years) under wear-levelling."""
    if write_bps <= 0:
        return float("inf")
    return n_f(T_C, n0, t0_C, ea_ev) * bits * wl / (write_bps * SECONDS_PER_YEAR)


def b_max_bytes_s(T_C, life_target_yr, *, n0=N0_CYCLES, t0_C=T0_C, ea_ev=EA_END_EV,
                  bits=BITS_PER_DIE, wl=WEAR_LEVEL_EFF):
    """Endurance-limited sustainable write bandwidth (byte/s per die) for a
    target service life -- the inverted form of life_years()."""
    return n_f(T_C, n0, t0_C, ea_ev) * bits * wl / (life_target_yr * SECONDS_PER_YEAR) / 8.0


# ── verification ─────────────────────────────────────────────────────────────
def verify():
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("verification")
    # 1. anchors + monotonicity
    chk("n_f(85 C) == N0", abs(n_f(T0_C) - N0_CYCLES) < 1e-3 * N0_CYCLES)
    chk("feram_retention_yr(85 C) == 10 yr", abs(feram_retention_yr(T0_C) - RET_YR_AT_T0) < 1e-9)
    chk("dram_retention_ms(85 C) == 64 ms", abs(dram_retention_ms(T0_C) - TREF_MS_AT_85) < 1e-9)
    temps = [60, 70, 80, 90, 100]
    chk("n_f decreasing in T", all(n_f(a) > n_f(b) for a, b in zip(temps, temps[1:])))
    chk("retention decreasing in T",
        all(feram_retention_yr(a) > feram_retention_yr(b) for a, b in zip(temps, temps[1:])))

    # 2. T_eff limit check + bracketing
    const = np.full(64, 88.0)
    chk("T_eff(constant) == constant", abs(t_eff_C(const, 0.5) - 88.0) < 1e-6)

    # 4. area assert against the floorplan
    flp = os.path.join(HERE, "5_thermal_si", "nv_tier_flp.csv")
    area = sum(l * w for n, _x, _y, l, w, _lab in _read_flp(flp) if n.startswith("MEM")) * 1e6
    chk("MEM area matches floorplan", abs(area - MEM_AREA_MM2) < 0.5,
        f"floorplan={area:.1f} mm^2, constant={MEM_AREA_MM2} mm^2")

    # 2b/3. bracketing + peak cross-check over the whole sweep
    ref = {}
    csv_path = os.path.join(SWEEP, "sweep_peaks.csv")
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path)):
            ref[int(r["n_feram"])] = (r["dram_peak_C"], r["feram_peak_C"])
    brack_ok, peak_ok, worst = True, True, 0.0
    for n in range(13):
        d = os.path.join(SWEEP, f"nv{n}")
        if not os.path.isdir(d):
            continue
        dies = die_temps(d)
        for _pos, _tech, pk, mn, cells in dies:
            for ea in set(EA_END_SWEEP) | set(EA_RET_SWEEP):
                te = t_eff_C(cells, ea)
                if not (mn - 1e-6 <= te <= pk + 1e-6):
                    brack_ok = False
        if n in ref:
            for tech, key in (("FeRAM", 1), ("DRAM", 0)):
                vals = [p for _, t, p, _, _ in dies if t == tech]
                if vals and ref[n][key]:
                    worst = max(worst, abs(max(vals) - float(ref[n][key])))
                    if abs(max(vals) - float(ref[n][key])) > 0.01:
                        peak_ok = False
    chk("mean <= T_eff <= peak (all dies, all Ea)", brack_ok)
    chk("peaks match sweep_peaks.csv", peak_ok, f"max deviation {worst:.4f} C")

    # 5. round-trip
    rt = life_years(94.7, b_max_bytes_s(94.7, 5.0) * 8.0)
    chk("round-trip b_max -> life == 5 yr", abs(rt - 5.0) < 1e-6, f"got {rt:.6f}")

    # 6. hand-checks at the bottom FeRAM die, N=4
    nf1 = n_f(94.7, ea_ev=0.5)
    yrs = life_years(94.7, write_bps_per_die(1.0), ea_ev=0.5)
    ret = feram_retention_yr(94.7, ea_ev=1.0)
    chk("N_f(94.7 C, Ea=0.5) ~ 6.5e11", abs(nf1 - 6.5e11) / 6.5e11 < 0.02, f"{nf1:.3e}")
    chk("life @100% writes ~ 6.6e3 yr", abs(yrs - 6.6e3) / 6.6e3 < 0.05, f"{yrs:.3e} yr")
    chk("FeRAM retention(94.7 C, Ea=1.0) ~ 4.26 yr", abs(ret - 4.26) < 0.05, f"{ret:.2f} yr")

    print("  ->", "ALL PASS" if ok else "FAILURES PRESENT")
    return ok


def _style(ax, xlabel, ylabel, ycolor=INK):
    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", color=ycolor)
    ax.grid(True, color=MUTE, alpha=0.25, lw=0.7, ls=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)
    ax.tick_params(labelsize=11.5, colors=MUTE)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)
        lbl.set_fontweight("bold")


def build_figure(grid_dir):
    """3-panel reliability figure for the XBM operating point."""
    dies = die_temps(grid_dir)
    feram = [(p, pk, mn, c) for p, t, pk, mn, c in dies if t == "FeRAM"]
    dram = [(p, pk, mn, c) for p, t, pk, mn, c in dies if t == "DRAM"]
    Ts = np.linspace(60.0, 100.0, 400)

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.0, 5.6), dpi=200)

    # ── (a) endurance, inverted: sustainable write bandwidth ─────────────────
    lo = [b_max_bytes_s(t, 10.0, ea_ev=min(EA_END_SWEEP)) / 1e9 for t in Ts]
    hi = [b_max_bytes_s(t, 10.0, ea_ev=max(EA_END_SWEEP)) / 1e9 for t in Ts]
    mid = [b_max_bytes_s(t, 10.0, ea_ev=EA_END_EV) / 1e9 for t in Ts]
    axA.fill_between(Ts, lo, hi, color=C_FERAM, alpha=0.18, zorder=2,
                     label=f"Ea {min(EA_END_SWEEP)}–{max(EA_END_SWEEP)} eV")
    axA.plot(Ts, mid, "-", color=C_FERAM, lw=2.6, zorder=5,
             label=f"$B_{{max}}$, 10-yr life (Ea {EA_END_EV} eV)")
    d_kv = write_bps_per_die(F_WRITE_KV) / 8.0 / 1e9
    d_wt = write_bps_per_die(F_WRITE_WEIGHTS) / 8.0 / 1e9
    axA.axhline(d_kv, ls="--", lw=2.0, color=C_DRAM, zorder=4,
                label=f"KV-cache write demand ({d_kv:.0f} GB/s)")
    axA.axhline(d_wt, ls="--", lw=2.0, color=C_TEAL, zorder=4,
                label=f"weight write demand ({d_wt*1000:.1f} MB/s)")
    # the stack's own FeRAM dies, on T_eff (wear-levelled ensemble)
    fx = [t_eff_C(c, EA_END_EV) for _p, _pk, _mn, c in feram]
    axA.plot(fx, [b_max_bytes_s(t, 10.0, ea_ev=EA_END_EV) / 1e9 for t in fx], "o",
             ms=9, mfc="white", mec=C_FERAM, mew=2.2, zorder=7, label="FeRAM dies (this stack)")
    margin = b_max_bytes_s(max(fx), 10.0, ea_ev=EA_END_EV) / 1e9 / d_kv
    axA.annotate(f"hottest FeRAM die:\n{margin:,.0f}× margin even\nvs KV-rate writes",
                 (max(fx), b_max_bytes_s(max(fx), 10.0, ea_ev=EA_END_EV) / 1e9),
                 xytext=(0.42, 0.30), textcoords="axes fraction",
                 fontsize=10.5, fontweight="bold", color=INK, zorder=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc=HL_FC, ec=HL_EC, lw=1.5),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                 connectionstyle="arc3,rad=0.2"))
    axA.set_yscale("log")
    _style(axA, "Die temperature (°C)", "Sustainable write BW (GB/s per die)", C_FERAM)
    axA.set_title("(a) Write endurance — huge margin",
                  fontsize=12.5, fontweight="bold", color=INK, pad=8)
    axA.legend(loc="lower left", frameon=True, framealpha=0.95,
               prop={"size": 8.6, "weight": "bold"})

    # ── (b) FeRAM retention — the binding constraint ─────────────────────────
    rlo = [feram_retention_yr(t, ea_ev=min(EA_RET_SWEEP)) for t in Ts]
    rhi = [feram_retention_yr(t, ea_ev=max(EA_RET_SWEEP)) for t in Ts]
    rmid = [feram_retention_yr(t, ea_ev=EA_RET_EV) for t in Ts]
    axB.fill_between(Ts, rlo, rhi, color=C_FERAM, alpha=0.18, zorder=2,
                     label=f"Ea {min(EA_RET_SWEEP)}–{max(EA_RET_SWEEP)} eV")
    axB.plot(Ts, rmid, "-", color=C_FERAM, lw=2.6, zorder=5,
             label=f"FeRAM retention (Ea {EA_RET_EV} eV)")
    for yr, lab in ((10.0, "10-yr service life"), (5.0, "5-yr")):
        axB.axhline(yr, ls=":", lw=1.9, color=C_DRAM, alpha=0.8, zorder=3)
        axB.annotate(lab, (99.6, yr), ha="right", va="bottom", fontsize=10,
                     fontweight="bold", color=C_DRAM, zorder=8)
    # retention fails at the hottest CELL, so peak (not T_eff) is the basis here
    px = [pk for _p, pk, _mn, _c in feram]
    axB.plot(px, [feram_retention_yr(t, ea_ev=EA_RET_EV) for t in px], "o", ms=9,
             mfc="white", mec=C_FERAM, mew=2.2, zorder=7, label="FeRAM dies (peak T)")
    r_hot, r_cool = feram_retention_yr(max(px), ea_ev=EA_RET_EV), feram_retention_yr(min(px), ea_ev=EA_RET_EV)
    axB.annotate(f"all 4 FeRAM dies fall SHORT of 10 yr\n"
                 f"{max(px):.1f} °C → {r_hot:.1f} yr   ·   {min(px):.1f} °C → {r_cool:.1f} yr",
                 (max(px), r_hot), xytext=(0.03, 0.13), textcoords="axes fraction",
                 fontsize=10.5, fontweight="bold", color=INK, zorder=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc=HL_FC, ec=HL_EC, lw=1.5),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                 connectionstyle="arc3,rad=-0.25"))
    axB.set_yscale("log")
    _style(axB, "Die temperature (°C)", "FeRAM retention (years)", C_FERAM)
    axB.set_title("(b) FeRAM retention — the real limit",
                  fontsize=12.5, fontweight="bold", color=INK, pad=8)
    axB.legend(loc="upper right", frameon=True, framealpha=0.95,
               prop={"size": 8.6, "weight": "bold"})

    # ── (c) DRAM retention — derives the 85/90 C limits ──────────────────────
    axC.plot(Ts, [dram_retention_ms(t) for t in Ts], "-", color=C_DRAM, lw=2.6,
             zorder=5, label=f"DRAM retention (÷2 per {HALVING_C:.0f} °C)")
    for T, lab, col, ha, dx in ((85.0, "85 °C", C_DRAM, "right", -0.6),
                                (90.0, "90 °C", "#e08214", "left", 0.6)):
        axC.axvline(T, ls=":", lw=1.9, color=col, alpha=0.85, zorder=3)
        axC.annotate(lab, (T + dx, 196), ha=ha, va="bottom", fontsize=10.5,
                     fontweight="bold", color=col, zorder=8)
    dx = [pk for _p, pk, _mn, _c in dram]
    axC.plot(dx, [dram_retention_ms(t) for t in dx], "s", ms=8, mfc="white",
             mec=C_DRAM, mew=2.0, zorder=7, label="DRAM dies (peak T)")
    axC.annotate(f"hottest DRAM die {max(dx):.1f} °C\n→ {dram_retention_ms(max(dx)):.0f} ms "
                 f"(vs {TREF_MS_AT_85:.0f} ms at 85 °C)",
                 (max(dx), dram_retention_ms(max(dx))), xytext=(0.05, 0.16),
                 textcoords="axes fraction", fontsize=10.5, fontweight="bold",
                 color=INK, zorder=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc=HL_FC, ec=HL_EC, lw=1.5),
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                 connectionstyle="arc3,rad=-0.2"))
    axC.set_ylim(0, 230)
    _style(axC, "Die temperature (°C)", "DRAM retention (ms)", C_DRAM)
    axC.set_title("(c) DRAM retention — origin of the limits",
                  fontsize=12.5, fontweight="bold", color=INK, pad=8)
    axC.legend(loc="upper right", frameon=True, framealpha=0.95,
               prop={"size": 8.6, "weight": "bold"})

    fig.text(0.5, 0.052,
             f"Endurance: N₀ = {N0_CYCLES:.0e} cycles @ {T0_C:.0f} °C, Ea {min(EA_END_SWEEP)}–{max(EA_END_SWEEP)} eV, "
             f"wear-levelled over {BITS_PER_DIE/8/1e9:.1f} GB/die "
             f"(0.45 Gb/mm² × {MEM_AREA_MM2:.0f} mm²); traffic {STACK_BW_TBS} TB/s/stack × α_w={ALPHA_W}, "
             f"F_write = {F_WRITE_WEIGHTS:g} (weights) / {F_WRITE_KV:g} (KV).",
             ha="center", va="bottom", fontsize=8.6, fontstyle="italic", color="#555555")
    fig.text(0.5, 0.020,
             "Endurance uses the wear-levelled ensemble temperature T_eff; retention uses peak T "
             "(the hottest cell loses data first). Temperatures measured from PACT; reliability modelled.",
             ha="center", va="bottom", fontsize=8.6, fontstyle="italic", color="#555555")

    fig.tight_layout(rect=(0, 0.085, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"nvdram_reliability.{ext}")
        fig.savefig(out, dpi=200)
        print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="run self-checks and exit")
    ap.add_argument("--run", default=os.path.join(SWEEP, "nv4"),
                    help="grid dir to analyse (default: the XBM N=4 operating point)")
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(0 if verify() else 1)
    build_figure(args.run)


if __name__ == "__main__":
    main()
