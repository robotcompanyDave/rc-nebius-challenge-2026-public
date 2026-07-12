#!/usr/bin/env python3
"""
Visual diagnostics for the Gate-M campaign (sweep2 / gate / gate2 JSONs).

Outputs into the given dir:
  arena_heatmap.png    16x16 grid of the 256 simultaneous envs, colored by
                       measured far-rim height — "the test arena"
  sweep_landscape.png  rim vs config, faceted by press depth & overhang
  bistability.png      gate2: same config x 8 scene slots — the pop-vs-swallow
                       spread that refuted the static 5mm objective
  gate_panel.png       the 8-jitter panel results for the sweep2 'winner'

    docker/run.sh tools/plot_gatem.py data/2026-07-05 OUTDIR
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = sys.argv[1] if len(sys.argv) > 1 else "data/2026-07-05"
outd = sys.argv[2] if len(sys.argv) > 2 else base
os.makedirs(outd, exist_ok=True)

sweep = json.load(open(os.path.join(base, "lab_sweep2", "sweep.json")))
gate = json.load(open(os.path.join(base, "lab_gate", "gate.json")))
gate2 = json.load(open(os.path.join(base, "lab_gate2", "gate2.json")))

# ── arena heatmap ────────────────────────────────────────────────────────────
N = len(sweep)
side = int(np.ceil(np.sqrt(N)))
grid = np.full((side, side), np.nan)
for r in sweep:
    e = r["env"]
    grid[e // side, e % side] = r["rim_mm"]
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=-5, vmax=10)
ax.set_title(f"The test arena — {N} simultaneous press tests "
             f"(one material×press system per cell)\n"
             f"color = far-rim height at hold (target 5 mm)")
ax.set_xlabel("env column (0.30 m spacing)")
ax.set_ylabel("env row")
cb = fig.colorbar(im, ax=ax, label="far-rim height (mm)")
cb.ax.axhline(5.0, color="k", lw=2)
tgt = [r for r in sweep if abs(r["rim_mm"] - 5) <= 1 and r["rim_std"] < 0.5]
for r in tgt:
    e = r["env"]
    ax.add_patch(plt.Rectangle((e % side - 0.5, e // side - 0.5), 1, 1,
                               fill=False, edgecolor="blue", lw=2.5))
fig.tight_layout()
fig.savefig(os.path.join(outd, "arena_heatmap.png"), dpi=130)
print("[plot] arena_heatmap.png")

# ── sweep landscape ──────────────────────────────────────────────────────────
presses = sorted({r["press"] for r in sweep})
overs = sorted({r["a_over"] for r in sweep})
fig, axs = plt.subplots(len(presses), len(overs), figsize=(13, 10),
                        sharex=True, sharey=True)
for pi, pd in enumerate(presses):
    for oi, ao in enumerate(overs):
        ax = axs[pi][oi]
        sel = [r for r in sweep if r["press"] == pd and r["a_over"] == ao]
        ks = sorted({r["stiffness"] for r in sel})
        for ratio, mk, col in ((3, "o", "#888"), (6, "s", "#2a6"),
                               (10, "^", "#26c")):
            xs = [r["stiffness"] for r in sel if r["ratio"] == ratio]
            ys = [r["rim_mm"] for r in sel if r["ratio"] == ratio]
            es = [2 * r["rim_std"] for r in sel if r["ratio"] == ratio]
            ax.errorbar(xs, ys, yerr=es, fmt=mk + "-", ms=5, color=col,
                        lw=1, label=f"ratio {ratio}" if (pi, oi) == (0, 0) else None)
        ax.axhspan(4, 6, color="#1e7a45", alpha=0.15)
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.set_title(f"press {1000*pd:.0f} mm · overhang {ao:.0%}", fontsize=10)
        ax.grid(alpha=0.25)
for ax in axs[-1]:
    ax.set_xlabel("k_cell (N/m)")
for row in axs:
    row[0].set_ylabel("rim (mm)")
fig.suptitle("Sweep 2 landscape — far-rim height vs material, per press geometry"
             " (green band = 5±1 mm target; bars = 2·std over hold)", fontsize=12)
fig.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(outd, "sweep_landscape.png"), dpi=130)
print("[plot] sweep_landscape.png")

# ── bistability (gate2 slots) ────────────────────────────────────────────────
rim = np.array(gate2["rim_mm"])
CFG = ["k300 r10 d12 p3 ao.70 (the 'winner')",
       "k300 r10 d24 p3 ao.70 (more damping)",
       "k420 r6 d8 p4 ao.50 (runner-up)",
       "k420 r6 d16 p4 ao.50 (escaped!)",
       "k300 r10 d20 p4 ao.60 (stable branch)"]
fig, ax = plt.subplots(figsize=(11, 6))
for ci in range(len(CFG)):
    block = rim[ci * 8:(ci + 1) * 8]
    block = np.clip(block, -12, 18)          # keep the flyer on-chart
    xs = np.full(8, ci) + np.linspace(-0.22, 0.22, 8)
    ax.scatter(xs, block, s=55, zorder=3,
               c=["#1e7a45" if abs(b - 5) <= 1 else "#b03a2e" for b in block])
    ax.hlines(block.mean(), ci - 0.3, ci + 0.3, color="k", lw=2.5, zorder=4)
ax.axhspan(4, 6, color="#1e7a45", alpha=0.15, label="5±1 mm target")
ax.axhline(0, color="#bbb", lw=0.8)
ax.set_xticks(range(len(CFG)))
ax.set_xticklabels([c.split(" (")[0] + "\n(" + c.split(" (")[1]
                    for c in CFG], fontsize=9)
ax.set_ylabel("far-rim height (mm)")
ax.set_title("The bistability that refuted static Gate M — SAME config, 8 scene"
             " slots each\n(dots = slots, bar = mean; k420 d16 clipped: washers"
             " flew, min −1218 mm)")
ax.legend()
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(outd, "bistability.png"), dpi=130)
print("[plot] bistability.png")

# ── gate jitter panel ────────────────────────────────────────────────────────
rimg = np.array(gate["rim_mm"])
jn = ["nominal", "x+1", "x-1", "y+1", "y-1", "z+.5", "z-.5", "combo"]
fig, ax = plt.subplots(figsize=(10, 5.5))
for rep in range(3):
    ax.plot(range(8), rimg[rep * 8:(rep + 1) * 8], "o--", ms=7,
            label=f"panel repeat {rep + 1}")
ax.axhspan(4, 6, color="#1e7a45", alpha=0.15, label="5±1 mm target")
ax.axhline(0, color="#bbb", lw=0.8)
ax.set_xticks(range(8))
ax.set_xticklabels(jn)
ax.set_ylabel("far-rim height (mm)")
ax.set_title("Sweep-2 'winner' on the 8-jitter panel ×3 — 0/24 in target band")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(outd, "gate_panel.png"), dpi=130)
print("[plot] gate_panel.png")
print("[plot] done")
