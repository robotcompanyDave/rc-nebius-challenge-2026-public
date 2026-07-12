#!/usr/bin/env python3
"""
Nebius Job entrypoint — PARALLEL grasp-attempt data generation.

K grasp cells in one physics scene (see graspsort/parallel_env.py). Each round runs
one grasp attempt in every cell simultaneously — the physics step is shared across
all K, so throughput scales ~K× vs the sequential jobs/gen_data.py, at the same
per-attempt fidelity (each cell runs the identical validated controller).

Config (env vars):
  GS_ENVS        parallel cells K                 (default 16)
  GS_N_ATTEMPTS  total grasp attempts             (default 512)
  GS_SEED        base RNG seed                    (default 0)
  GS_OUTPUT_DIR  dataset dir / s3://bucket/prefix (default ./data/dataset_par)
  GS_HEADLESS    1 headless (default), 0 GUI

Output identical to gen_data.py: records.jsonl (+ records.parquet) via DatasetWriter.
One SimulationApp for the whole run; the dataset flushes to the sink every round, so
a mid-run crash still leaves every completed round persisted.
"""
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENVS = int(os.environ.get("GS_ENVS", "16"))
N_ATTEMPTS = int(os.environ.get("GS_N_ATTEMPTS", "512"))
SEED = int(os.environ.get("GS_SEED", "0"))
OUT_DIR = os.environ.get("GS_OUTPUT_DIR", os.path.join("data", "dataset_par"))
HEADLESS = os.environ.get("GS_HEADLESS", "1") != "0"
MAX_TICKS = 3000


def _clutter(ctrl, target, paths):
    """Scene clutter around the target (mirror of gen_data._clutter)."""
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


def main():
    import numpy as np
    from graspsort.parallel_env import ParallelGraspEnv
    from graspsort.controller import GraspSortController
    from graspsort import randomize
    from graspsort.logging_schema import DatasetWriter, AttemptRecord
    from graspsort.parts import PartSpec

    writer = DatasetWriter(OUT_DIR)
    print(f"[par] envs={ENVS} target={N_ATTEMPTS} seed={SEED} out={OUT_DIR}", flush=True)

    env = ParallelGraspEnv(n_envs=ENVS, headless=HEADLESS)
    env.reset_world()
    ctrls = [GraspSortController(env.cell_view(i)) for i in range(ENVS)]
    centres = [c.work_centre_xy() for c in ctrls]
    print(f"[par] booted {ENVS} cells; cell-0 work centre {centres[0]}", flush=True)

    done = 0
    rnd = 0
    while done < N_ATTEMPTS:
        k = min(ENVS, N_ATTEMPTS - done)          # active cells this round
        # v2 scenes: multi-part bundles per cell; specs sampled at the CELL's
        # work centre (sample_scene_multi xys are already absolute).
        cell_specs, cell_paths, targets = [], [], []
        for i in range(k):
            rng = random.Random(SEED * 1_000_003 + rnd * 10_007 + i)
            specs = randomize.sample_scene_multi(rng, centres[i])
            paths = env.set_cell_parts(i, specs)
            t_idx = int(np.argmin([np.hypot(s.xy[0] - centres[i][0],
                                            s.xy[1] - centres[i][1]) for s in specs]))
            cell_specs.append(specs); cell_paths.append(paths)
            targets.append(t_idx)
        env.settle(40)

        # begin a grasp on every cell's target part
        cands, scene0s = [], []
        for i in range(k):
            rng = random.Random(SEED * 1_000_003 + rnd * 10_007 + i + 500_009)
            path = cell_paths[i][targets[i]]
            spec = cell_specs[i][targets[i]]
            scene0 = _clutter(ctrls[i], path, cell_paths[i])
            heur = ctrls[i].grasp_R(path, spec.kind)
            h_yaw = float(np.arctan2(heur[1, 0], heur[0, 0]))
            cand = randomize.sample_candidate(rng, spec.kind, heuristic_yaw=h_yaw,
                                              cluttered=scene0["n_close"] > 0)
            cand["heuristic_yaw"] = h_yaw
            cands.append(cand); scene0s.append(scene0)
            ctrls[i].reset_to_home()
            ctrls[i].begin_grasp(path, spec.kind, cand)

        # drive all active cells together; ONE physics step per tick
        for _tick in range(MAX_TICKS):
            busy = False
            for i in range(k):
                if ctrls[i].busy:
                    ctrls[i].step(ctrls[i].dt)
                    busy = True
            env.step(render=False)
            if not busy:
                break
        for i in range(k):
            ctrls[i].force_finish()

        # record every cell's outcome (v2 schema — mirrors gen_data.py, no cameras)
        for i in range(k):
            o = ctrls[i].outcome
            spec = cell_specs[i][targets[i]]
            path = cell_paths[i][targets[i]]
            cand = cands[i]
            scene0 = scene0s[i]
            scene1 = _clutter(ctrls[i], path, cell_paths[i])
            aid = f"{rnd:04d}_{i:03d}"
            writer.write(AttemptRecord(
                attempt_id=aid, seed=SEED, batch=rnd,
                obs={
                    "topdown_rgb": None, "topdown_depth": None,
                    "eih_rgb": None, "eih_depth": None,
                    "part": {"kind": spec.kind, "size": spec.size, "pose": spec.pose,
                             "xy": list(spec.xy), "rotz_deg": spec.rotz_deg},
                    "scene": {"n_parts": len(cell_paths[i]), **scene0},
                },
                action={
                    "grasp_yaw": cand["grasp_yaw"], "xy_offset": list(cand["xy_offset"]),
                    "grasp_dz": cand["grasp_dz"], "approach_dh": cand["approach_dh"],
                    "width": cand["width"],
                    "strategy": cand.get("strategy", "direct"),
                    "tilt_deg": cand.get("tilt_deg", 0.0),
                    "pre_drag": bool(cand.get("pre_drag")),
                    "heuristic_yaw": cand["heuristic_yaw"],
                },
                outcome={
                    "success": o.success, "lifted_mm": o.lifted_mm,
                    "grasp_force_N": o.grasp_force_N, "clamp_openness": o.clamp_openness,
                    "slip_mm": o.slip_mm, "fail_reason": o.fail_reason,
                    "nearest_mm_after": scene1["nearest_mm"],
                    "separation_mm": (
                        None if (scene0["nearest_mm"] is None or scene1["nearest_mm"] is None)
                        else scene1["nearest_mm"] - scene0["nearest_mm"]),
                }))
        done += k
        rnd += 1
        writer.to_parquet()
        writer.flush_to_final()
        print(f"[par] round {rnd}: {done}/{N_ATTEMPTS} attempts ({writer.n} records persisted)",
              flush=True)

    writer.close()
    writer.flush_to_final()
    print(f"[par] DONE — {writer.n} records → {OUT_DIR}/records.jsonl", flush=True)
    env.close()


if __name__ == "__main__":
    main()
