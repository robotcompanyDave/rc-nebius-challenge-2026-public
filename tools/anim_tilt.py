#!/usr/bin/env python3
"""
Render a 2D side-profile ANIMATION of a soft_tilt descent, straight from the
measured per-tick data (descent.jsonl) — no Isaac, no camera. Faithful: the
washer bar uses the measured near/far rim heights, the surface dip uses the
measured max deflection. Shows the whole behaviour: gentle tilt in the 10-20
band, then the tip-over past the cliff.

    python3 tools/anim_tilt.py <descent.jsonl> [out.mp4]
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

R = 12.0            # washer OD/2 mm
THK = 2.5          # washer thickness mm
PRESS_X = -6.0     # press offset (mm), matches GS_ST_OFFSET default
BAND = (10.0, 20.0)


def surface_y(x, defl):
    """Deformed surface height (mm, <=0) — flat, dipping to -defl around the
    press point, recovering on the far side (a cosine bowl)."""
    y = np.zeros_like(x)
    lo, hi = PRESS_X - 8.0, PRESS_X + 12.0
    m = (x > lo) & (x < hi)
    t = (x[m] - lo) / (hi - lo)             # 0..1 across the bowl
    y[m] = -defl * (0.5 - 0.5 * np.cos(2 * math.pi * np.clip(t, 0, 1))) \
        * (1.0 - np.clip((x[m] - PRESS_X) / (hi - PRESS_X), 0, 1) * 0.2)
    return y


def frame(rec):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.set_xlim(-22, 22); ax.set_ylim(-17, 22)
    ax.set_aspect("equal"); ax.axis("off")
    ax.axhspan(-17, 0, color="#eef1f4", zorder=0)
    # target band shading on the far-edge side (visual reference only)
    # rest surface
    ax.axhline(0, color="#9aa2ad", ls="--", lw=1)
    ax.text(21.5, 0.6, "rest surface", ha="right", fontsize=8, color="#5b6470")
    # deformed surface
    xs = np.linspace(-22, 22, 260)
    ys = surface_y(xs, rec["defl_mm"])
    ax.plot(xs, ys, color="#2fb3a3", lw=3, zorder=2)
    ax.fill_between(xs, ys, -17, color="#d9f2ee", zorder=1)
    # washer bar from near rim to far rim: heights are measured, and the rim
    # X-positions compress by cos(tilt) so the bar length/angle stays correct
    # even as it stands up (fixed +/-R would draw a vertical washer at ~45deg).
    nz, fz = rec["near_z"], rec["far_z"]
    rx = R * math.cos(math.radians(rec["tilt"]))
    nx, fx = -rx, rx
    dxu = np.array([fx - nx, fz - nz]); dxu = dxu / (np.linalg.norm(dxu) + 1e-9)
    perp = np.array([-dxu[1], dxu[0]]) * (THK / 2)
    poly = np.array([[nx + perp[0], nz + perp[1]], [fx + perp[0], fz + perp[1]],
                     [fx - perp[0], fz - perp[1]], [nx - perp[0], nz - perp[1]]])
    ax.add_patch(plt.Polygon(poly, closed=True, facecolor="#c73a2e",
                             edgecolor="#7a1f16", lw=1.4, zorder=4))
    # finger pressing the near side
    pd = rec.get("press_mm", 0.0)
    fb = -pd                                  # finger bottom (below surface)
    ax.add_patch(plt.Rectangle((PRESS_X - 2.5, fb), 5.0, 20, facecolor="#3a3f46",
                               edgecolor="none", alpha=0.82, zorder=5))
    ax.plot([PRESS_X - 2.5, PRESS_X + 2.5], [fb, fb], color="#2f6fe0", lw=3, zorder=6)
    # labels
    tilt = rec["tilt"]
    incol = "#157a4f" if BAND[0] <= tilt <= BAND[1] else "#b3382e"
    ax.text(-21, 19, f"press {pd:4.1f} mm", fontsize=12, color="#1f2733")
    ax.text(-21, 15.5, f"tilt {tilt:4.1f}°", fontsize=13, color=incol, fontweight="bold")
    ax.text(21, 19, "compliant spring surface (k=300)", ha="right", fontsize=9,
            color="#5b6470")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/soft_tilt/descent.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(src), "tilt_anim.mp4")
    recs = [json.loads(l) for l in open(src)]
    press = [r for r in recs if r.get("phase") == "press"]
    if not press:
        print("no press records"); return
    frames = []
    for r in press:
        frames.append(frame(r))
        frames.append(frames[-1])            # 2x for a slower, readable pace
    # hold the last frame
    for _ in range(20):
        frames.append(frames[-1])
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
