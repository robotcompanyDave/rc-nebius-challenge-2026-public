#!/usr/bin/env python3
"""Camera sanity probe: boot env, spawn one washer at the work centre, render
with (a) the exact proven ReviewCam geometry and (b) the close-up press cam,
saving frames at several warmup depths. Diagnoses the uniform-grey captures."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get("GS_CAM_OUT", "/tmp/claude-1000/-home-david-RC/98e5fbe8-2991-4d27-a5fe-d8356d70e2a7/scratchpad/camtest")


def main():
    import numpy as np
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort import parts

    env = GraspSortEnv(headless=True)
    # isaacsim modules are importable only after the SimulationApp exists
    from render_review import look_at_quat
    from isaacsim.sensors.camera import Camera
    import cv2
    env.reset_world()
    ctrl = GraspSortController(env)
    wc = ctrl.work_centre_xy()
    surf = env.table_top_z
    env.spawn_parts([parts.PartSpec(kind="washer", size="m12", pose="flat",
                                    xy=wc, rotz_deg=0.0)])
    env.settle(40)

    cams = {}
    eyeA = np.array([wc[0] + 0.62, wc[1] - 0.72, surf + 0.58])
    tgtA = np.array([wc[0], wc[1], surf + 0.04])
    cams["review34"] = Camera(prim_path="/World/CamA", position=eyeA,
                              frequency=30, resolution=(960, 540),
                              orientation=look_at_quat(eyeA, tgtA))
    eyeB = np.array([wc[0] + 0.16, wc[1] - 0.30, surf + 0.16])
    tgtB = np.array([wc[0], wc[1], surf + 0.01])
    cams["press_close"] = Camera(prim_path="/World/CamB", position=eyeB,
                                 frequency=30, resolution=(960, 540),
                                 orientation=look_at_quat(eyeB, tgtB))
    for c in cams.values():
        c.initialize()

    os.makedirs(OUT, exist_ok=True)
    for warm in (5, 15, 40):
        for _ in range(warm):
            env.step(render=True)
        for name, c in cams.items():
            rgba = c.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                print(f"[cam] {name} warm{warm}: EMPTY", flush=True)
                continue
            rgb = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            print(f"[cam] {name} warm{warm}: mean={rgb.mean():.1f} std={rgb.std():.1f}",
                  flush=True)
            cv2.imwrite(os.path.join(OUT, f"{name}_w{warm}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[cam] DONE -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
