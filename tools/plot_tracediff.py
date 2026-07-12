#!/usr/bin/env python3
"""Task #23: same-θ trace diff, pick_lab vs full env (GraspSortEnv).

Left: washer z (mm rel surface) both harnesses; middle: tilt; right: the
DIFFS over the common time window. The divergence onset time names the
culprit phase (settle → bed mounting; press → finger z reference;
close/rise → B spring rates).

    docker/run.sh tools/plot_tracediff.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = os.path.join(BASE, "data/2026-07-05/tracediff_env/trace.json")
LAB = os.path.join(BASE, "data/2026-07-05/tracediff_lab/results.json")
OUT = os.path.join(BASE, "data/2026-07-05/charts/tracediff.png")

env = np.array(json.load(open(ENV)))            # (t, vc, tilt, wz_mm)
res = json.load(open(LAB))
mat = next(iter(res))
g = res[mat]["gauntlet"][0]["panel"][0]         # cand0, rig0 (zero jitter)
lab = np.array(g["trace"])

# align on common timestamps (lab sampling starts after settle; env at 0)
et = {round(r[0], 2): r for r in env}
lt = {round(r[0], 2): r for r in lab}
common = sorted(set(et) & set(lt))
env = np.array([et[t] for t in common])
lab = np.array([lt[t] for t in common])
n = len(common)
te = tl = env[:, 0]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
axes[0].plot(te, env[:n, 3], label="env wz", color="#d62728")
axes[0].plot(tl, lab[:n, 3], label="lab wz", color="#1f77b4")
axes[0].set_title("washer z (mm rel surface)")
axes[1].plot(te, env[:n, 2], label="env tilt", color="#d62728")
axes[1].plot(tl, lab[:n, 2], label="lab tilt", color="#1f77b4")
axes[1].set_title("washer tilt (deg)")
dz = env[:n, 3] - lab[:n, 3]
dt_ = env[:n, 2] - lab[:n, 2]
axes[2].plot(te, dz, label="Δz (env−lab) mm", color="#2ca02c")
axes[2].plot(te, dt_ / 10, label="Δtilt/10 (deg)", color="#9467bd")
axes[2].axhline(0, color="k", lw=0.5)
axes[2].set_title(f"diffs — Δz@rest={dz[:8].mean():+.2f}mm, "
                  f"Δz@end={dz[-8:].mean():+.2f}mm")
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_xlabel("t (s)")
fig.suptitle(f"lab ↔ env same-θ trace diff (cand0, {mat}, zero jitter)")
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=110)
print(f"wrote {OUT}")
print(f"[diff] rest Δz={dz[:8].mean():+.3f}mm  press Δz="
      f"{dz[(te>1.8)&(te<2.8)].mean():+.3f}mm  "
      f"end Δz={dz[-8:].mean():+.3f}mm  max|Δtilt|={np.abs(dt_).max():.1f}deg "
      f"onset t={te[np.argmax(np.abs(dz)>0.5)] if (np.abs(dz)>0.5).any() else -1:.2f}s")
