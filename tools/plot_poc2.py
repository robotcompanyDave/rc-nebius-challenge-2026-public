#!/usr/bin/env python3
"""FEM POC-2 chart: far-rim trajectories, 2 E-values x 4 pad positions.
    docker/run.sh tools/plot_poc2.py   (or any python with matplotlib)"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data/2026-07-05/poc_fem_press/poc2.json")
OUT = os.path.join(BASE, "data/2026-07-05/charts/poc2_press.png")

recs = json.load(open(SRC))
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, young in zip(axes, (0.2, 0.3)):
    ax.axhspan(4, 6, color="#b7e4c7", alpha=0.8, zorder=0)
    for r in recs:
        if r["young_mpa"] != young:
            continue
        tr = np.array(r["traj"])
        t = 0.5 + np.arange(len(tr)) * 0.05
        ax.plot(t, tr, lw=1.4,
                label=f"shift ({r['shift_mm'][0]:.1f},{r['shift_mm'][1]:.1f})mm"
                      f"  hold {r['hold_mean']:+.1f}±{r['hold_std']:.1f}")
    ax.axvline(1.7, color="k", lw=0.6, ls=":")
    ax.set_title(f"E = {young} MPa")
    ax.set_xlabel("t (s)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
axes[0].set_ylabel("far-rim height vs surface (mm)")
fig.suptitle("FEM POC-2 — canonical press on deformable pad, "
             "4 sub-cell positions (gate: smooth + position-invariant)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
