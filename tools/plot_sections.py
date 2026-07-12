#!/usr/bin/env python3
"""
To-scale material CROSS-SECTIONS at end-of-press, parameters annotated.
Reads sections.json from a gate2/gate3 run; draws, per selected env:
tile columns (center row), the washer (true tilt from its quaternion),
both fingers of the bound gripper, rest-surface & target lines.

    docker/run.sh tools/plot_sections.py data/2026-07-05/lab_gate3 OUTDIR
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

base = sys.argv[1]
outd = sys.argv[2] if len(sys.argv) > 2 else base
os.makedirs(outd, exist_ok=True)
S = json.load(open(os.path.join(base, "sections.json")))

W_R, W_T = 12.0, 2.5     # mm
FW, FH = 8.0, 30.0
CELL = 5.0


def tilt_deg(q):
    w, x, y, z = q
    # local +z after rotation; tilt = angle of washer plane vs horizontal
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1, min(1, abs(zz)))))


def tilt_signed(q):
    """Rotation angle about the y-axis (the press axis is x)."""
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * y + x * z),
                                   1 - 2 * (y * y + x * x)))


def draw(env, ax):
    m = env["mat"]
    tiles = env["tiles"]
    for tx, tz in tiles:                       # tz = tile TOP height (mm)
        ax.add_patch(Rectangle((tx - CELL * 0.46, tz - 10), CELL * 0.92, 10,
                               facecolor="#e2d0ac", edgecolor="#8a6d3b",
                               lw=0.5))
    # washer: rectangle OD x thickness rotated by its true pitch
    wx, wz = env["washer"]
    ang = tilt_signed(env["quat"])
    ax.add_patch(Rectangle((wx - W_R, wz - W_T / 2), 2 * W_R, W_T,
                           angle=ang, rotation_point="center",
                           facecolor="#e0654a", edgecolor="#8a2f1d", lw=1))
    # fingers (bound pair) at press depth
    a_c, pz, g0 = env["a_c_mm"], env["press_z_mm"], env["g0_mm"]
    for fx, col in ((a_c, "#39424e"), (a_c + g0, "#4a90c4")):
        ax.add_patch(Rectangle((fx - FW / 2, pz), FW, FH,
                               facecolor=col, edgecolor="none", alpha=0.9))
    ax.axhline(0, color="#999", ls="--", lw=0.9)
    ax.axhline(5, color="#1e7a45", ls=":", lw=1.2)
    ax.text(26, 5.5, "5mm target", fontsize=7, color="#1e7a45")
    rim = wz + W_R * math.sin(math.radians(abs(ang)))  # coarse rim marker
    ax.set_xlim(-30, 32)
    ax.set_ylim(-18, 26)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.set_title(f"env {env['env']}  —  tilt {ang:+.1f}°, washer z {wz:+.1f} mm",
                 fontsize=9)
    ax.text(0.02, 0.02,
            f"k={m['stiffness']:.0f} N/m  ratio={m['ratio']:.0f} "
            f"(couple {m['couple']:.0f})\n"
            f"damping={m['damping']:.0f}  press={1000*m['press']:.0f} mm  "
            f"overhang={m['a_over']:.0%}",
            transform=ax.transAxes, fontsize=7.5, family="monospace",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999"))


# pick per config: the most-lifted and least-lifted slot (pop vs swallow)
byc = {}
for e in S:
    key = (e["mat"]["stiffness"], e["mat"]["ratio"], e["mat"]["damping"],
           e["mat"]["press"], e["mat"]["a_over"])
    byc.setdefault(key, []).append(e)

names = list(byc.keys())
fig, axs = plt.subplots(len(names), 2, figsize=(11, 3.1 * len(names)))
for ci, key in enumerate(names):
    envs = byc[key]
    envs.sort(key=lambda e: e["washer"][1] + W_R * math.sin(
        math.radians(abs(tilt_signed(e["quat"])))))
    draw(envs[0], axs[ci][0])
    draw(envs[-1], axs[ci][1])
    axs[ci][0].set_ylabel("z (mm)")
axs[0][0].set_title("LOWEST slot (swallow branch)\n" + axs[0][0].get_title(),
                    fontsize=9)
axs[0][1].set_title("HIGHEST slot (pop branch)\n" + axs[0][1].get_title(),
                    fontsize=9)
for a in axs[-1]:
    a.set_xlabel("x (mm) — A presses left rim, B waits right")
fig.suptitle("Material cross-sections at END OF PRESS — same config, two scene"
             " slots (the bistability, in section)", fontsize=12)
fig.tight_layout()
fp = os.path.join(outd, "cross_sections.png")
fig.savefig(fp, dpi=130)
print(f"[plot] {fp}")
