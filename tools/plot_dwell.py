#!/usr/bin/env python3
"""Gate M' dwell charts from lab_dwell1/dwell.json + lab_gate4/gate2.json.

Left: rim trajectories for the 7 gate4 candidates x 8 slots (one panel per
candidate, 4-6mm band shaded) -- shows park vs transit vs stay-down per slot.
Right: dwell_ms per slot bars.

Runs on host (numpy+matplotlib only).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G4 = os.path.join(BASE, "data/2026-07-05/lab_gate4/gate2.json")
OUT = os.path.join(BASE, "data/2026-07-05/charts")

CANDS = [
    ("k300 r10 d12 p3 ao.70  (sweep 900ms slot)", "#1f77b4"),
    ("k300 r10 d24 p3 ao.70  (parker)", "#2ca02c"),
    ("k420 r6 d8 p4 ao.50", "#ff7f0e"),
    ("k420 r6 d16 p4 ao.50", "#d62728"),
    ("k300 r10 d20 p4 ao.60", "#9467bd"),
    ("k220 r10 d12 p4 ao.70  (transiter)", "#8c564b"),
    ("k300 r3 d8 p3 ao.70", "#7f7f7f"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(G4))
    D = np.array(d["traj"])              # (S, 56) rim mm, 20 Hz from T_SET
    dwell = np.array(d["dwell_ms"])
    S = D.shape[0]
    t = 0.5 + np.arange(S) * 0.05        # sample clock (T_SET + k/20Hz)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axes = axes.ravel()
    for ci, (label, color) in enumerate(CANDS):
        ax = axes[ci]
        ax.axhspan(4, 6, color="#b7e4c7", alpha=0.8, zorder=0,
                   label="4-6mm meet band")
        for s in range(8):
            e = ci * 8 + s
            ax.plot(t, D[:, e], color=color, alpha=0.55, lw=1.2)
        ax.axvline(1.7, color="k", lw=0.6, ls=":")   # press bottom-out
        ax.set_ylim(-8, 25)
        ax.set_title(f"{label}\ndwell/slot: "
                     f"{[int(v) for v in dwell[ci*8:(ci+1)*8]]} ms",
                     fontsize=9)
        ax.grid(alpha=0.25)
        if ci >= 4:
            ax.set_xlabel("t (s)")
        if ci % 4 == 0:
            ax.set_ylabel("far-rim height vs surface (mm)")
    # last panel: dwell bars
    ax = axes[7]
    for ci, (label, color) in enumerate(CANDS):
        ax.bar(np.arange(8) + ci * 0.11 - 0.35,
               dwell[ci * 8:(ci + 1) * 8], width=0.1, color=color,
               label=label.split("  ")[0])
    ax.axhline(150, color="r", lw=0.8, ls="--")
    ax.set_xlabel("scene slot")
    ax.set_ylabel("band dwell (ms)")
    ax.set_title("dwell per slot per candidate (red: 150ms floor)",
                 fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Gate M' — far-rim trajectories during press+hold, "
                 "7 candidates x 8 scene slots (CPU, canonical press)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "dwell_slots.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
