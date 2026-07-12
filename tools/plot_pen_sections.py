#!/usr/bin/env python3
"""Cross-sections of the FEM press from poc_fem_pen profiles_*.json:
pad top center-row + washer rectangle at press milestones.
    docker/run.sh tools/plot_pen_sections.py"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data/2026-07-05/poc_fem_pen")
OUT = os.path.join(BASE, "data/2026-07-05/charts/fem_pen_sections.png")

files = sorted(glob.glob(os.path.join(SRC, "profiles_*.json")))
fig, axes = plt.subplots(len(files), 5, figsize=(22, 3.2 * len(files)),
                         sharex=True, sharey=True)
if len(files) == 1:
    axes = axes[None, :]
for r, f in enumerate(files):
    tag = os.path.basename(f)[9:-5]
    profs = json.load(open(f))
    for c, pr in enumerate(profs[:5]):
        ax = axes[r][c]
        x = np.array(pr["x"]) * 1000
        z = np.array(pr["z"])
        ax.fill_between(x, -10, z, color="#e8d5b5", zorder=1)
        ax.plot(x, z, color="#b08d55", lw=1.5, zorder=2)
        w = np.array(pr["washer"] + [pr["washer"][0]])
        ax.fill(w[:, 0] * 1000, w[:, 1], color="#d94f3d", alpha=0.9,
                zorder=3)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlim(-45, 45)
        ax.set_ylim(-11, 8)
        if r == 0:
            ax.set_title(pr["label"], fontsize=10)
        if c == 0:
            ax.set_ylabel(f"{tag}\nz (mm)", fontsize=9)
        ax.grid(alpha=0.2)
fig.suptitle("FEM press cross-sections (pad top center row + washer) — "
             "how the washer enters the pad", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
