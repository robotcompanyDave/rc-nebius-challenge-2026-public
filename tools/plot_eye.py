#!/usr/bin/env python3
"""
Eye diagrams for the pick maneuver: washer TILT (vertical target = 90°) vs
FINGER CLOSURE fraction, all 8 jitter-panel runs overlaid per candidate + the
mean line — the spread of the "eye" reads as smoothness/consistency the same
way a telecom eye diagram does.

Input: a pick_lab results.json whose gauntlet was run with GS_PL_TRACE=1.
Output: eye_cand{N}.png per candidate (+ eye_all.png 2x2 composite).

    docker/run.sh tools/plot_eye.py data/2026-07-04/lab_eye/results.json OUTDIR
(no Isaac needed — plain matplotlib; runs in the container for the deps)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res_path = sys.argv[1] if len(sys.argv) > 1 else "data/2026-07-04/lab_eye/results.json"
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(res_path)
res = json.load(open(res_path))
mname = next(iter(res))
gaunt = sorted(res[mname]["gauntlet"], key=lambda g: g["cand"])

JN = ["nominal", "x+1", "x-1", "y+1", "y-1", "z+.5", "z-.5", "combo"]
GRID = np.linspace(0.0, 1.0, 80)

figs = []
ncand = len(gaunt)
nrow = (ncand + 1) // 2
fig_all, axs = plt.subplots(nrow, 2, figsize=(13, 4.5 * nrow),
                            sharex=True, sharey=True)
axs = np.atleast_2d(axs)
for gi, g in enumerate(gaunt):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    interps = []
    for r, p in enumerate(g["panel"]):
        tr = p.get("trace", [])
        if not tr:
            continue
        vc = np.array([s[1] for s in tr])
        tl = np.array([s[2] for s in tr])
        ok = p.get("success")
        col = "#1e7a45" if ok else "#b03a2e"
        for a in (ax, axs.flat[gi]):
            a.plot(vc, tl, "-", lw=1.1, alpha=0.45, color=col,
                   label=(JN[r % len(JN)] if (a is ax and r < len(JN)) else None))
        m = vc > 1e-4                      # post-grip, closure engaged
        if m.sum() > 3:
            v, t_ = vc[m], tl[m]
            keep = np.concatenate([[True], np.diff(v) > 1e-6])
            interps.append(np.interp(GRID, v[keep], t_[keep],
                                     left=np.nan, right=np.nan))
    if interps:
        M = np.nanmean(np.vstack(interps), axis=0)
        for a in (ax, axs.flat[gi]):
            a.plot(GRID, M, "-", lw=3.0, color="#1a1a2e", label="mean")
    for a in (ax, axs.flat[gi]):
        a.axhline(90, ls="--", lw=1, color="#888")
        a.axhline(55, ls=":", lw=1, color="#bb8")
        a.set_xlim(-0.02, 1.02)
        a.set_ylim(-5, 130)
        a.grid(alpha=0.25)
    ttl = (f"cand{g['cand']} — robust {g['robust']:.2f}, "
           f"{g['succ']}/{g['n']} on panel")
    ax.set_title(f"Eye diagram: {ttl}")
    ax.set_xlabel("finger closure fraction (0 = at grip trigger, 1 = closed)")
    ax.set_ylabel("washer tilt (°)  —  90 = vertical target")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    axs.flat[gi].set_title(ttl, fontsize=10)
    fig.tight_layout()
    fp = os.path.join(outdir, f"eye_cand{g['cand']}.png")
    fig.savefig(fp, dpi=130)
    figs.append(fp)
    print(f"[eye] wrote {fp}")

for a in axs[-1]:
    a.set_xlabel("finger closure fraction")
for a in axs[:, 0]:
    a.set_ylabel("tilt (°)")
for k in range(ncand, axs.size):
    axs.flat[k].axis("off")
fig_all.suptitle(f"{mname} — tilt vs closure, 8-jitter panel per candidate "
                 f"(green=success, red=fail, bold=mean)", fontsize=12)
fig_all.tight_layout()
fp = os.path.join(outdir, "eye_all.png")
fig_all.savefig(fp, dpi=130)
print(f"[eye] wrote {fp}")
