#!/usr/bin/env python3
"""
Headless REVIEW render — boot the grasp+sort env, spawn a small scene, run one
full pick→place, and capture a 3/4 camera (stills + MP4) so a human can eyeball
what the sim looks like. NOT part of the Nebius Job pipeline — a dev/review tool.

Renders are written to $GS_REVIEW_DIR (default ./data/review, gitignored). Uses an
`isaacsim.sensors.camera.Camera` render product (works headless on this box; the
full GUI viewport does not — see the Optimus/Blackwell note). Steps WITH rendering
on, with warm-up frames, so unlike the data-gen obs cameras the frames aren't None.

    "D:/isaacsim/python.bat" tools/render_review.py
Env knobs: GS_REVIEW_DIR, GS_REVIEW_PARTS, GS_REVIEW_EVERY (capture cadence),
GS_REVIEW_W/H, GS_REVIEW_MODE (pickplace|grasp), GS_REVIEW_SEED, and
GS_CAM_EYE / GS_CAM_TGT as "x,y,z" to override the camera.
"""
import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from graspsort.videoio import to_h264

OUT = os.environ.get("GS_REVIEW_DIR", os.path.join("data", "review"))
N_PARTS = int(os.environ.get("GS_REVIEW_PARTS", "5"))
EVERY = int(os.environ.get("GS_REVIEW_EVERY", "3"))
W = int(os.environ.get("GS_REVIEW_W", "960"))
H = int(os.environ.get("GS_REVIEW_H", "540"))
MODE = os.environ.get("GS_REVIEW_MODE", "pickplace")
SEED = int(os.environ.get("GS_REVIEW_SEED", "7"))
MAX_TICKS = 4000


def _vec_env(name):
    v = os.environ.get(name)
    if not v:
        return None
    return np.array([float(x) for x in v.split(",")], dtype=float)


def look_at_quat(eye, target, world_up=(0.0, 0.0, 1.0)):
    """(w,x,y,z) world quat for an isaacsim Camera (optical axis = local +X, up = +Z)
    looking from `eye` toward `target`."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    f = target - eye
    f /= (np.linalg.norm(f) + 1e-12)                  # local +X -> forward
    wu = np.asarray(world_up, float)
    u = wu - np.dot(wu, f) * f
    if np.linalg.norm(u) < 1e-6:                       # looking straight up/down
        u = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], f) * f
    u /= (np.linalg.norm(u) + 1e-12)                   # local +Z -> up
    y = np.cross(u, f)                                 # local +Y
    R = np.column_stack([f, y, u])
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s
        yq = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            yq = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            yq = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            yq = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, yq, z])


def _prettify(env):
    """Review-only cosmetics (does NOT touch the data-gen env): tame the blown-out
    exposure by lowering the lights and darkening the big gray surfaces so the steel
    parts + gripper read with contrast."""
    from pxr import UsdGeom, UsdLux
    st = env.stage

    def set_intensity(path, val, kind):
        p = st.GetPrimAtPath(path)
        if p and p.IsValid():
            kind(p).GetIntensityAttr().Set(val)

    set_intensity("/World/DomeLight", 350.0, UsdLux.DomeLight)
    set_intensity("/World/KeyLight", 800.0, UsdLux.DistantLight)

    def set_color(path, rgb):
        p = st.GetPrimAtPath(path)
        if p and p.IsValid():
            g = UsdGeom.Gprim(p)
            (g.GetDisplayColorAttr() or g.CreateDisplayColorAttr()).Set([tuple(rgb)])

    set_color("/World/SortPlatform", (0.16, 0.18, 0.23))   # cool dark slab
    set_color("/World/Ground", (0.09, 0.09, 0.10))


def main():
    frames_dir = os.path.join(OUT, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort import randomize, parts as P

    env = GraspSortEnv(headless=True, width=W, height=H)
    env.reset_world()
    _prettify(env)                      # review-only lighting/surface tweaks
    ctrl = GraspSortController(env)
    wc = ctrl.work_centre_xy()
    rng = random.Random(SEED)

    # ── spawn a small mixed scene around the work centre ──────────────────────
    kinds = ["nut", "bolt", "nut", "bolt", "nut", "bolt"]
    poses = ["flat", "standing", "random", "on-side", "flat", "random"]
    sizes = ["m12", "m8", "m16", "m12", "m8", "m12"]
    slots = P.scatter_xy(wc, N_PARTS, spread=0.07, min_sep=0.05, rng=rng)
    specs = [P.PartSpec(kind=kinds[i % 6], size=sizes[i % 6], pose=poses[i % 6],
                        xy=xy, rotz_deg=rng.uniform(0.0, 360.0))
             for i, xy in enumerate(slots)]
    env.clear_parts()
    paths = env.spawn_parts(specs)
    env.settle(90)

    # resting-pose check: lowest point of each part's LOAD-BEARING geometry vs the
    # table (for a bolt, the shank). A naturally resting part is ~0 mm; the old box
    # collider floated a lying bolt's shank ~half-a-head (several mm) above the table.
    from pxr import UsdGeom as _UG
    _bb = _UG.BBoxCache(0, [_UG.Tokens.default_, _UG.Tokens.render])
    for pth, sp in zip(paths, specs):
        sub = pth + ("/shank" if sp.kind == "bolt" else "/body")
        pr = env.stage.GetPrimAtPath(sub)
        if pr and pr.IsValid():
            box = _bb.ComputeWorldBound(pr).ComputeAlignedBox()
            zmin = float(box.GetMin()[2])
            print(f"[rest] {sp.kind:4} {sp.pose:8} {sp.size}: "
                  f"lowest_pt = {1000.0 * (zmin - env.table_top_z):+.1f} mm vs table",
                  flush=True)

    # ── review camera (3/4 view), overridable via env ────────────────────────
    target = _vec_env("GS_CAM_TGT")
    if target is None:
        target = np.array([wc[0], wc[1], env.table_top_z + 0.04])
    eye = _vec_env("GS_CAM_EYE")
    if eye is None:
        eye = np.array([wc[0] + 0.62, wc[1] - 0.72, env.table_top_z + 0.58])

    from isaacsim.sensors.camera import Camera
    cam = Camera(prim_path="/World/ReviewCam", position=eye, frequency=30,
                 resolution=(W, H), orientation=look_at_quat(eye, target))
    cam.initialize()
    for _ in range(15):
        env.step(render=True)

    import cv2

    def grab():
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    def save(name, rgb):
        if rgb is not None:
            cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    frames = []

    def cap(every_ok=True):
        fr = grab()
        if fr is not None and every_ok:
            frames.append(fr)
            cv2.imwrite(os.path.join(frames_dir, f"f{len(frames):04d}.png"),
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        return fr

    save("00_overview.png", grab())

    ctrl.reset_to_home()
    for _ in range(10):
        env.step(render=True)
    save("01_home.png", cap())

    # ── run one full pick→place (or grasp) on the first part ─────────────────
    tgt, spec0 = paths[0], specs[0]
    heur = ctrl.grasp_R(tgt, spec0.kind)
    h_yaw = float(np.arctan2(heur[1, 0], heur[0, 0]))
    cand = randomize.sample_candidate(rng, spec0.kind, heuristic_yaw=h_yaw)
    if MODE == "grasp":
        ctrl.begin_grasp(tgt, spec0.kind, cand)
    else:
        ctrl.begin_pick_place(tgt, cand)

    n = 0
    last_state = None
    while ctrl.busy and n < MAX_TICKS:
        ctrl.step(ctrl.dt)
        env.step(render=(n % EVERY == 0))
        if n % EVERY == 0:
            cap()
        if ctrl.state != last_state:           # a still at each phase change
            last_state = ctrl.state
            save(f"phase_{last_state}.png", grab())
        n += 1
    ctrl.force_finish()
    for _ in range(8):
        env.step(render=True)
    save("02_final.png", cap())

    o = ctrl.outcome
    print(f"[review] mode={MODE} success={o.success} lifted_mm={o.lifted_mm:.1f} "
          f"force_N={o.grasp_force_N:.1f} frames={len(frames)} ticks={n}", flush=True)

    if frames:
        h, w = frames[0].shape[:2]
        mp4 = os.path.join(OUT, "grasp_review.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        to_h264(mp4)
        print(f"[review] wrote {mp4} ({len(frames)} frames, {w}x{h})", flush=True)
    print(f"[review] stills + frames in {OUT}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
