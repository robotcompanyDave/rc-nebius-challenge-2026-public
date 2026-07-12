#!/usr/bin/env python3
"""
Nebius Job entrypoint — grasp-attempt data generation.

Runs N domain-randomized single-part grasp attempts headless and writes a
labeled `(observation, candidate grasp, success)` dataset to $GS_OUTPUT_DIR
(an S3 bucket mounted into the Job). Each attempt:

  spawn 1 random M-part → settle → drive approach+descend → CAPTURE pre-grasp
  observation → close (force-feedback) → lift → label by whether it held.

Isaac cold start is amortized by running GS_BATCH attempts per SimulationApp; a
fresh app is booted per batch as a stability valve (the platform's nutsandbolts
pipeline hit Replicator/heap drift past ~60 frames in one app).

Config (env vars):
  GS_SEED        base RNG seed                (default 0)
  GS_N_ATTEMPTS  total grasp attempts         (default 200)
  GS_BATCH       attempts per SimulationApp   (default 50)
  GS_OUTPUT_DIR  dataset output dir / bucket  (default ./data/dataset)
  GS_HEADLESS    1 headless (default), 0 GUI
"""
import os
import sys
import random

# package import (jobs/ sits next to graspsort/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = int(os.environ.get("GS_SEED", "0"))
N_ATTEMPTS = int(os.environ.get("GS_N_ATTEMPTS", "200"))
BATCH = int(os.environ.get("GS_BATCH", "50"))
OUT_DIR = os.environ.get("GS_OUTPUT_DIR", os.path.join("data", "dataset"))
HEADLESS = os.environ.get("GS_HEADLESS", "1") != "0"
# GS_OBS=0 → skip camera capture entirely. The supervised grasp-SUCCESS scorer is
# trained on STATE (part kind/size/pose + candidate grasp) → success, so it needs
# no images; skipping them makes the cloud data-gen Job much faster and avoids the
# camera-annotator warm-up issue. Set GS_OBS=1 only for the (future) vision branch.
CAPTURE_OBS = os.environ.get("GS_OBS", "0") != "0"

MAX_TICKS = 3000
_NULL_OBS = {"topdown_rgb": None, "topdown_depth": None,
             "eih_rgb": None, "eih_depth": None}


def _clutter(ctrl, target, paths):
    """Scene clutter around the target: neighbour count within 40 mm + nearest (mm)."""
    import numpy as np
    pose = ctrl.part_world_pose(target)
    if pose is None:
        return {"n_close": 0, "nearest_mm": None}
    cx, cy = float(pose[0, 3]), float(pose[1, 3])
    n_close, nearest = 0, None
    for p in paths:
        if p == target:
            continue
        pp = ctrl.part_world_pose(p)
        if pp is None:
            continue
        d = float(np.hypot(float(pp[0, 3]) - cx, float(pp[1, 3]) - cy))
        if d < 0.04:
            n_close += 1
        nearest = d if nearest is None else min(nearest, d)
    return {"n_close": n_close,
            "nearest_mm": None if nearest is None else nearest * 1000.0}


def run_chunk(batch_idx, n, writer, env, ctrl, work_centre, observer):
    """Run `n` grasp attempts on an ALREADY-BOOTED env (no boot/teardown here).
    Isaac's SimulationApp.close() hard-exits the process, so main() boots the sim
    once and closes it last — see the note in main().

    v2: multi-part scenes (bundles ~40%), washers + m6–m12 sizes, and the pick
    STRATEGY (direct/tilt/lip, optional pre-drag) in the action. Labels carry the
    grasp success plus the separation effect (nearest-neighbour distance before →
    after) so drag-separation is learnable alongside the grasp."""
    from graspsort import randomize
    from graspsort.logging_schema import AttemptRecord

    done = 0
    for i in range(n):
        rng = random.Random(SEED * 1_000_003 + batch_idx * 10_007 + i)
        specs = randomize.sample_scene_multi(rng, work_centre)
        env.clear_parts()
        paths = env.spawn_parts(specs)
        env.settle(40)
        # target = the part nearest the work centre (deterministic given the seed)
        import numpy as np
        t_idx = int(np.argmin([np.hypot(s.xy[0] - work_centre[0],
                                        s.xy[1] - work_centre[1]) for s in specs]))
        path, spec = paths[t_idx], specs[t_idx]

        scene0 = _clutter(ctrl, path, paths)
        heuristic = ctrl.grasp_R(path, spec.kind)
        # heuristic yaw about world Z extracted from the grasp_R for perturbation
        h_yaw = float(np.arctan2(heuristic[1, 0], heuristic[0, 0]))
        cand = randomize.sample_candidate(rng, spec.kind, heuristic_yaw=h_yaw,
                                          cluttered=scene0["n_close"] > 0)

        ctrl.reset_to_home()
        ctrl.begin_grasp(path, spec.kind, cand)
        obs_imgs = None
        n_tick = 0
        while ctrl.busy and n_tick < MAX_TICKS:
            ctrl.step(ctrl.dt)
            env.step(render=False)
            if observer is not None and obs_imgs is None and ctrl.state == "grasp":
                obs_imgs = observer.capture()        # pre-grasp: at descend, jaws open
            n_tick += 1
        ctrl.force_finish()
        if obs_imgs is None:
            obs_imgs = observer.capture() if observer is not None else dict(_NULL_OBS)
        outcome = ctrl.outcome
        scene1 = _clutter(ctrl, path, paths)         # separation label (post-attempt)

        aid = f"{batch_idx:03d}_{i:05d}"
        rec = AttemptRecord(
            attempt_id=aid, seed=SEED, batch=batch_idx, ts_step=n_tick,
            obs={
                "topdown_rgb": writer.save_image(f"{aid}_top_rgb.png", obs_imgs["topdown_rgb"]),
                "topdown_depth": writer.save_image(f"{aid}_top_d.png", obs_imgs["topdown_depth"]),
                "eih_rgb": writer.save_image(f"{aid}_eih_rgb.png", obs_imgs["eih_rgb"]),
                "eih_depth": writer.save_image(f"{aid}_eih_d.png", obs_imgs["eih_depth"]),
                "part": {"kind": spec.kind, "size": spec.size, "pose": spec.pose,
                         "xy": list(spec.xy), "rotz_deg": spec.rotz_deg},
                "scene": {"n_parts": len(paths), **scene0},
            },
            action={
                "grasp_yaw": cand["grasp_yaw"], "xy_offset": list(cand["xy_offset"]),
                "grasp_dz": cand["grasp_dz"], "approach_dh": cand["approach_dh"],
                "width": cand["width"],
                "strategy": cand.get("strategy", "direct"),
                "tilt_deg": cand.get("tilt_deg", 0.0),
                "pre_drag": bool(cand.get("pre_drag")),
                # heuristic (expert) yaw this candidate perturbs around — lets the
                # scorer train on the yaw DELTA from the heuristic, which is the
                # frame-invariant signal (absolute world yaw is not learnable).
                "heuristic_yaw": h_yaw,
            },
            outcome={
                "success": outcome.success, "lifted_mm": outcome.lifted_mm,
                "grasp_force_N": outcome.grasp_force_N,
                "clamp_openness": outcome.clamp_openness, "slip_mm": outcome.slip_mm,
                "fail_reason": outcome.fail_reason,
                "nearest_mm_after": scene1["nearest_mm"],
                "separation_mm": (
                    None if (scene0["nearest_mm"] is None or scene1["nearest_mm"] is None)
                    else scene1["nearest_mm"] - scene0["nearest_mm"]),
            },
        )
        writer.write(rec)
        done += 1
        if done % 10 == 0:
            print(f"[gen] batch {batch_idx} attempt {i + 1}/{n} "
                  f"success={outcome.success} strat={cand.get('strategy')} "
                  f"(total {writer.n})", flush=True)

    return done


def main():
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort.observe import Observer
    from graspsort.logging_schema import DatasetWriter

    writer = DatasetWriter(OUT_DIR)
    print(f"[gen] seed={SEED} target={N_ATTEMPTS} batch={BATCH} out={OUT_DIR}", flush=True)

    # ONE SimulationApp for the whole run. The per-batch "fresh app" valve is gone:
    # GS_OBS=0 removes the Replicator/camera churn that motivated it, and Isaac's
    # SimulationApp.close() HARD-EXITS the process — so any code after a close()
    # (parquet roll-up, bucket flush) would never run, and a 2nd app can't boot
    # in-process. We therefore flush to the bucket after each chunk and close LAST.
    env = GraspSortEnv(headless=HEADLESS)
    env.reset_world()
    ctrl = GraspSortController(env)
    work_centre = ctrl.work_centre_xy()
    observer = None
    if CAPTURE_OBS:
        observer = Observer(env.robot_cfg["tool_link"], work_centre, env.table_top_z)
        observer.setup()

    done, batch_idx = 0, 0
    while done < N_ATTEMPTS:
        n = min(BATCH, N_ATTEMPTS - done)
        run_chunk(batch_idx, n, writer, env, ctrl, work_centre, observer)
        done += n
        batch_idx += 1
        writer.to_parquet()            # refresh the parquet roll-up
        writer.flush_to_final()        # persist progress to the (S3) bucket NOW
        print(f"[gen] chunk {batch_idx} done; {done}/{N_ATTEMPTS} attempts "
              f"({writer.n} records persisted)", flush=True)

    pq = writer.to_parquet()
    writer.close()
    writer.flush_to_final()
    print(f"[gen] DONE - {writer.n} records -> {OUT_DIR}/records.jsonl"
          + (" + records.parquet" if pq else ""), flush=True)
    env.close()                        # LAST — may hard-exit; data already flushed


if __name__ == "__main__":
    main()
