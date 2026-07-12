#!/usr/bin/env python3
"""
Headless HEURISTIC-vs-SCORER sort comparison render (the "does training pay off"
visual). Boots the grasp+sort env ONCE, then runs the SAME mixed nut/bolt sort
scene twice — once with the hand-tuned heuristic grasp ("before"), once with the
trained scorer picking the best of N candidate grasps ("after") — capturing a 3/4
camera each time. Writes per-policy MP4s + a side-by-side compare.mp4 with a live
"sorted N/M" overlay. NOT part of the Nebius Job pipeline; a dev/review tool.

    "D:/isaacsim/python.bat" tools/render_sort.py
Env knobs: GS_REVIEW_DIR (default data/review_sort), GS_MODEL (scorer .npz,
default data/model/model.npz), GS_REVIEW_PARTS, GS_REVIEW_EVERY, GS_REVIEW_W/H,
GS_REVIEW_SEED, GS_CANDS.
"""
import os
import sys
import random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))       # repo root (graspsort package)
sys.path.insert(0, _HERE)                         # for render_review helpers

import numpy as np
from render_review import look_at_quat, _prettify
from graspsort.videoio import to_h264

OUT = os.environ.get("GS_REVIEW_DIR", os.path.join("data", "review_sort"))
MODEL = os.environ.get("GS_MODEL", os.path.join("data", "model", "model.npz"))
N_PARTS = int(os.environ.get("GS_REVIEW_PARTS", "6"))
EVERY = int(os.environ.get("GS_REVIEW_EVERY", "3"))
W = int(os.environ.get("GS_REVIEW_W", "960"))
H = int(os.environ.get("GS_REVIEW_H", "540"))
SEED = int(os.environ.get("GS_REVIEW_SEED", "7"))
N_CANDS = int(os.environ.get("GS_CANDS", "24"))
MAX_TICKS = 4000
SORT_MAX_ATTEMPTS = 3


HARD = os.environ.get("GS_REVIEW_HARD", "0") != "0"


def build_scene(env, ctrl, rng):
    """Mixed nut/bolt/washer scene — same recipe as jobs/eval_sort.build_scene,
    covering all THREE sort lanes (nuts −DY, bolts +DY, washers +2·DY). With
    GS_REVIEW_HARD=1, bias toward the hard grasp cases (flat/random nuts, the
    M6-washer-flat training target) so the heuristic-vs-scorer difference is
    visible."""
    from graspsort import parts
    if HARD:
        scenarios = [("nut", "flat", "m12"), ("nut", "random", "m12"),
                     ("washer", "flat", "m6"), ("nut", "random", "m12"),
                     ("washer", "flat", "m12"), ("nut", "flat", "m12")]
    else:
        scenarios = [("bolt", "standing", "m12"), ("nut", "on-side", "m12"),
                     ("washer", "flat", "m12"), ("nut", "flat", "m12"),
                     ("bolt", "on-side", "m12"), ("washer", "on-side", "m12")]
    centre = ctrl.work_centre_xy()
    xys = parts.scatter_xy(centre, N_PARTS, spread=0.06, min_sep=0.05, rng=rng)
    specs = []
    for i in range(N_PARTS):
        if i < len(scenarios):
            kind, pose, size = scenarios[i]
        else:
            kind = rng.choice(parts.KINDS)
            pose = rng.choice(parts.POSE_CLASSES)
            size = "m12"
        specs.append(parts.PartSpec(kind=kind, size=size, pose=pose, xy=xys[i],
                                    rotz_deg=rng.uniform(0, 360)))
    env.clear_parts()
    env.spawn_parts(specs)
    env.settle(60)


def make_scorer_policy(ctrl):
    """Argmax-over-candidates scorer policy (same logic as eval_sort)."""
    from graspsort import randomize
    from graspsort.scorer import GraspScorer
    if not os.path.isfile(MODEL):
        print(f"[sort-render] no model at {MODEL}; scorer pass will fall back to heuristic",
              flush=True)
        return None
    scorer = GraspScorer.load(MODEL)
    print(f"[sort-render] scorer loaded (val AUC={scorer.meta.get('val_auc')})", flush=True)
    rng = random.Random(SEED * 104729 + 1)

    def policy(part_path, c):
        spec = c.env.part_specs.get(part_path)
        kind = spec.kind if spec else c.env.part_kinds.get(part_path, "nut")
        size = spec.size if spec else "m12"
        pose = spec.pose if spec else "flat"
        R = c.grasp_R(part_path, kind)
        h_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        part = {"kind": kind, "size": size, "pose": pose}
        cands = [{"xy_offset": (0.0, 0.0), "grasp_yaw": h_yaw, "grasp_dz": 0.0,
                  "approach_dh": 0.12, "heuristic_yaw": h_yaw, "width": 1.0}]
        for _ in range(N_CANDS):
            cc = randomize.sample_candidate(rng, kind, heuristic_yaw=h_yaw)
            cc["heuristic_yaw"] = h_yaw
            cands.append(cc)
        probs = scorer.score_batch(part, cands)
        return cands[int(probs.argmax())]
    return policy


def _correct_count(env, ctrl):
    n = 0
    for p in env.part_paths:
        pose = ctrl.part_world_pose(p)
        if pose is None:
            continue
        kind = env.part_kinds.get(p, "nut")
        want = "nuts" if kind == "nut" else "bolts"
        if ctrl.part_zone(float(pose[0, 3]), float(pose[1, 3])) == want:
            n += 1
    return n


def run_sort_render(env, ctrl, cam, policy, label, out_sub, cv2):
    """Run one full sort with rendering; return the list of RGB frames."""
    frames_dir = os.path.join(OUT, out_sub, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    n_parts = len(env.part_paths)

    def grab():
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            return None
        return np.asarray(rgba)[:, :, :3].astype(np.uint8)

    frames = []

    def overlay(fr, sorted_n):
        fr = fr.copy()
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 38), (24, 24, 28), -1)
        cv2.putText(fr, f"{label}", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (240, 240, 255), 2, cv2.LINE_AA)
        txt = f"sorted {sorted_n}/{n_parts}"
        cv2.putText(fr, txt, (fr.shape[1] - 190, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (140, 240, 140), 2, cv2.LINE_AA)
        return fr

    def cap():
        fr = grab()
        if fr is None:
            return
        fr = overlay(fr, _correct_count(env, ctrl))
        frames.append(fr)
        cv2.imwrite(os.path.join(frames_dir, f"f{len(frames):04d}.png"),
                    cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))

    ctrl.reset_to_home()
    for _ in range(10):
        env.step(render=True)
    cap()

    picks, attempts, skip = 0, {}, set()
    while picks < 12:
        cands = [p for p in ctrl._sort_candidates() if p not in skip]
        if not cands:
            break
        part = cands[0]
        if attempts.get(part, 0) >= SORT_MAX_ATTEMPTS:
            skip.add(part)
            continue
        attempts[part] = attempts.get(part, 0) + 1
        cand = policy(part, ctrl) if policy else None
        ctrl.begin_pick_place(part, cand)
        n = 0
        while ctrl.busy and n < MAX_TICKS:
            ctrl.step(ctrl.dt)
            env.step(render=(n % EVERY == 0))
            if n % EVERY == 0:
                cap()
            n += 1
        ctrl.force_finish()
        ctrl.reset_to_home()
        for k in range(18):
            env.step(render=(k % EVERY == 0))
            if k % EVERY == 0:
                cap()
        picks += 1

    correct = _correct_count(env, ctrl)
    print(f"[sort-render] {label}: sorted {correct}/{n_parts} in {picks} picks, "
          f"{len(frames)} frames", flush=True)
    # a couple of held final frames
    for _ in range(10):
        frames.append(overlay(grab() if grab() is not None else frames[-1], correct))
    # per-policy mp4
    if frames:
        h, w = frames[0].shape[:2]
        mp4 = os.path.join(OUT, f"{out_sub}.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        to_h264(mp4)
        print(f"[sort-render] wrote {mp4}", flush=True)
    return frames, correct


def main():
    os.makedirs(OUT, exist_ok=True)
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    import cv2

    env = GraspSortEnv(headless=True, width=W, height=H)
    env.reset_world()
    _prettify(env)
    ctrl = GraspSortController(env)
    wc = ctrl.work_centre_xy()

    target = np.array([wc[0], wc[1], env.table_top_z + 0.04])
    eye = np.array([wc[0] + 0.62, wc[1] - 0.72, env.table_top_z + 0.58])
    from isaacsim.sensors.camera import Camera
    cam = Camera(prim_path="/World/ReviewCam", position=eye, frequency=30,
                 resolution=(W, H), orientation=look_at_quat(eye, target))
    cam.initialize()
    for _ in range(15):
        env.step(render=True)

    scorer_policy = make_scorer_policy(ctrl)

    # heuristic ("before") on the seeded scene
    build_scene(env, ctrl, random.Random(SEED))
    fr_h, c_h = run_sort_render(env, ctrl, cam, None, "HEURISTIC (before)", "heuristic", cv2)

    # scorer ("after") on the SAME seeded scene
    build_scene(env, ctrl, random.Random(SEED))
    fr_s, c_s = run_sort_render(env, ctrl, cam, scorer_policy, "SCORER (after)", "scorer", cv2)

    # side-by-side compare.mp4 (pad the shorter with its last frame)
    m = max(len(fr_h), len(fr_s))
    if fr_h and fr_s:
        fr_h += [fr_h[-1]] * (m - len(fr_h))
        fr_s += [fr_s[-1]] * (m - len(fr_s))
        gap = np.full((fr_h[0].shape[0], 6, 3), 40, np.uint8)
        h, w = fr_h[0].shape[0], fr_h[0].shape[1] * 2 + 6
        mp4 = os.path.join(OUT, "compare.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for a, b in zip(fr_h, fr_s):
            comb = np.concatenate([a, gap, b], axis=1)
            vw.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
        vw.release()
        to_h264(mp4)
        print(f"[sort-render] wrote {mp4}  (heuristic {c_h} vs scorer {c_s} of {N_PARTS})",
              flush=True)
    env.close()


if __name__ == "__main__":
    main()
