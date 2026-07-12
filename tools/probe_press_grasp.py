#!/usr/bin/env python3
"""
Press-lift STAGE 2 probe — the full washer pick via the new `press_lift`
controller strategy: servo one finger onto the near rim, press+hold (the pad
pivots the washer, stage-1-proven), coordinated close (TCP shifts so the press
finger stays pinned; the closing finger sweeps under the raised far rim), lift.

Runs N single-washer attempts (m12 flat + m6 flat) and prints per-attempt
outcomes with the per-phase geometry. Run with the stage-1 material recipe:

    GS_SOFT_SPREAD=1.0 ~/TOOLS/isaac-sim/python.sh tools/probe_press_grasp.py

Env: GS_PG_OUT (default data/<date>/<HHMM>-press_grasp), GS_PG_N (default 4
per size), GS_PG_SIZES (default m12,m6).
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_now = datetime.datetime.now()
OUT = os.environ.get("GS_PG_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-press_grasp"))
N_PER = int(os.environ.get("GS_PG_N", "4"))
SIZES = os.environ.get("GS_PG_SIZES", "m12,m6").split(",")


CAM = os.environ.get("GS_PG_CAM", "1") != "0"


def main():
    import random
    import numpy as np
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort import parts

    env = GraspSortEnv(headless=True)
    env.reset_world()
    ctrl = GraspSortController(env)
    assert env.soft_rig is not None
    centre = ctrl.work_centre_xy()
    print(f"[pg] spread_gain={env.soft_rig.pad.spread_gain} "
          f"depth={env.soft_rig.depth * 1000:.0f}mm", flush=True)

    os.makedirs(OUT, exist_ok=True)

    # side-on camera perpendicular to the press direction (u = +X): shows the
    # press finger going down and the far rim lifting, in profile.
    cam = None
    cv2 = None
    if CAM:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from render_review import look_at_quat
        from isaacsim.sensors.camera import Camera
        import cv2 as _cv2
        cv2 = _cv2
        surf = env.table_top_z
        # the PROVEN ReviewCam 3/4 angle (probe_cam.py: a tight low view sees
        # only featureless slab), at full and half distance
        tgt = (centre[0], centre[1], surf + 0.02)
        cams = []
        for name, eye in (
                ("far", (centre[0] + 0.62, centre[1] - 0.72, surf + 0.58)),):
            cc = Camera(prim_path=f"/World/PressCam_{name}",
                        position=np.array(eye), frequency=30,
                        resolution=(960, 540),
                        orientation=look_at_quat(eye, tgt))
            cc.initialize()
            cams.append((name, cc))
        cam = cams
        for _ in range(15):
            env.step(render=True)
        os.makedirs(os.path.join(OUT, "shots"), exist_ok=True)

    def snap(tag):
        if cam is None:
            return
        for _ in range(3):
            env.step(render=True)
        for name, cc in cam:
            rgba = cc.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                continue
            rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, "shots", f"{tag}_{name}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    VIDEO = os.environ.get("GS_PG_VIDEO", "0") != "0"

    def attempt_with_shots(ai, path):
        """attempt_grasp with per-phase stills and (GS_PG_VIDEO=1) an MP4."""
        ctrl.begin_grasp(path, "washer", {"strategy": "press_lift",
                                          "lead_dir": (-1.0, 0.0)})
        last_state, n = None, 0
        frames = []

        def frame(label):
            if cam is None:
                return
            env.step(render=True)
            rgba = cam[0][1].get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                return
            fr = np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()
            cv2.putText(fr, f"{label}  t={n}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2,
                        cv2.LINE_AA)
            frames.append(fr)

        while ctrl.busy and n < 4000:
            ctrl.step(ctrl.dt)
            env.step(render=False)
            n += 1
            if ctrl.state != last_state:
                snap(f"a{ai}_t{n:04d}_{ctrl.state}")
                last_state = ctrl.state
            elif n % 40 == 0 and ctrl.state.startswith(("pw_", "lift")):
                snap(f"a{ai}_t{n:04d}_{ctrl.state}")
            if VIDEO and n % 6 == 0:
                frame(ctrl.state)
        ctrl.force_finish()
        snap(f"a{ai}_t{n:04d}_end")
        if VIDEO and frames:
            vp = os.path.join(OUT, f"press_{ai}.mp4")
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 18,
                                 (frames[0].shape[1], frames[0].shape[0]))
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"[pg] wrote {vp} ({len(frames)} frames)", flush=True)
        return ctrl.outcome

    results = []
    ai = 0
    for size in SIZES:
        for i in range(N_PER):
            rng = random.Random(1000 + i)
            spec = parts.PartSpec(kind="washer", size=size, pose="flat",
                                  xy=(centre[0] + rng.uniform(-0.01, 0.01),
                                      centre[1] + rng.uniform(-0.01, 0.01)),
                                  rotz_deg=rng.uniform(0, 360))
            env.clear_parts()
            env.spawn_parts([spec])
            env.settle(50)
            path = env.part_paths[0]
            ctrl.reset_to_home()
            for _ in range(30):
                env.step(render=False)
            out = attempt_with_shots(f"{size}_{i}", path)
            ai += 1
            rec = {"size": size, "i": i,
                   "success": out.success, "lifted_mm": round(out.lifted_mm, 1),
                   "force_N": round(out.grasp_force_N, 1),
                   "clamp": round(out.clamp_openness, 2),
                   "slip_mm": round(out.slip_mm, 1), "fail": out.fail_reason}
            results.append(rec)
            print(f"[pg] {size} #{i}: success={out.success} "
                  f"lift={out.lifted_mm:.0f}mm force={out.grasp_force_N:.0f}N "
                  f"clamp={out.clamp_openness:.2f} fail={out.fail_reason}",
                  flush=True)

    ok = {s: sum(r["success"] for r in results if r["size"] == s) for s in SIZES}
    summary = {"per_size": {s: f"{ok[s]}/{N_PER}" for s in SIZES},
               "spread_gain": env.soft_rig.pad.spread_gain, "results": results}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[pg] DONE {summary['per_size']} -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
