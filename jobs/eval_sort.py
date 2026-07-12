#!/usr/bin/env python3
"""
Nebius Job entrypoint — sort-success evaluation (the headline metric).

Runs M multi-part nut/bolt sort trials and reports the fraction of parts landed
in their correct zone, under one or both policies:

  • heuristic  — the ported hand-tuned grasp (controller.grasp_R); the "before".
  • scorer     — sample candidate grasps, score each with a trained grasp-success
                 model (ONNX), execute the argmax; the "after".

Writes $GS_OUTPUT_DIR/report.json with mean ± std success per policy and a
per-trial / per-part breakdown.

Config (env vars):
  GS_SEED        base seed                          (default 0)
  GS_TRIALS      number of sort trials              (default 10)
  GS_PARTS       parts per trial                    (default 6)
  GS_POLICY      heuristic | scorer | both          (default heuristic)
  GS_MODEL       path to grasp-scorer ONNX (scorer) (default "")
  GS_OUTPUT_DIR  report output dir / bucket         (default ./data/eval)
  GS_HEADLESS    1 headless (default), 0 GUI
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = int(os.environ.get("GS_SEED", "0"))
TRIALS = int(os.environ.get("GS_TRIALS", "10"))
PARTS = int(os.environ.get("GS_PARTS", "6"))
POLICY = os.environ.get("GS_POLICY", "heuristic").lower()
MODEL = os.environ.get("GS_MODEL", "").strip()
OUT_DIR = os.environ.get("GS_OUTPUT_DIR", os.path.join("data", "eval"))
HEADLESS = os.environ.get("GS_HEADLESS", "1") != "0"


def build_scene(env, ctrl, rng):
    """A mixed nut/bolt/washer scene scattered on the platform. First parts are
    the deterministic M12 scenarios (vert bolt / nut-side / WASHER-flat / nut-flat
    — mirrors the platform's SORT_SCENARIOS on feat/ur10e-soft-sort) for
    repeatability; the rest are random over the m6–m12 working span."""
    from graspsort import parts
    scenarios = [("bolt", "standing"), ("nut", "on-side"),
                 ("washer", "flat"), ("nut", "flat")]
    centre = ctrl.work_centre_xy()
    xys = parts.scatter_xy(centre, PARTS, rng=rng)
    specs = []
    for i in range(PARTS):
        if i < len(scenarios):
            kind, pose = scenarios[i]
            size = "m12"
        else:
            kind = rng.choice(parts.KINDS)
            pose = rng.choice(parts.POSE_CLASSES)
            size = rng.choice(parts.WORK_SIZES)
        specs.append(parts.PartSpec(kind=kind, size=size, pose=pose, xy=xys[i],
                                    rotz_deg=rng.uniform(0, 360)))
    env.clear_parts()
    env.spawn_parts(specs)
    env.settle(60)


N_CANDS = int(os.environ.get("GS_CANDS", "24"))


def _part_state(ctrl, part_path):
    """(kind, size, pose) for a part, from the env's stored PartSpec (falls back to
    kind-only if unknown). This is the STATE half of the scorer's input."""
    spec = ctrl.env.part_specs.get(part_path)
    if spec is not None:
        return spec.kind, spec.size, spec.pose
    return ctrl.env.part_kinds.get(part_path, "nut"), "m12", "flat"


def _heuristic_yaw(ctrl, part_path, kind):
    import numpy as np
    R = ctrl.grasp_R(part_path, kind)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def make_scorer_policy(model_path):
    """policy(part_path, ctrl)->candidate that samples N_CANDS grasps and executes
    the one the trained scorer rates most likely to HOLD. The pure heuristic grasp is
    always in the candidate set, so the policy can never (by its own belief) do worse
    than the heuristic — it only deviates when it predicts a higher success. Falls
    back to heuristic (None) if the model can't be loaded."""
    import random
    from graspsort import randomize
    from graspsort.scorer import GraspScorer
    if not model_path or not os.path.isfile(model_path):
        print(f"[eval] no scorer model at '{model_path}'; using heuristic", flush=True)
        return None
    try:
        scorer = GraspScorer.load(model_path)
    except Exception as e:
        print(f"[eval] scorer load failed ({e}); using heuristic", flush=True)
        return None
    print(f"[eval] scorer loaded ({model_path}); val AUC={scorer.meta.get('val_auc')}, "
          f"{N_CANDS} candidates/pick", flush=True)
    rng = random.Random(SEED * 104729 + 1)

    def policy(part_path, ctrl):
        import numpy as np
        kind, size, pose = _part_state(ctrl, part_path)
        h_yaw = _heuristic_yaw(ctrl, part_path, kind)
        part = {"kind": kind, "size": size, "pose": pose}
        # scene clutter for the scorer's state (n_close / nearest_mm)
        pose_m = ctrl.part_world_pose(part_path)
        n_close, nearest = 0, None
        if pose_m is not None:
            cx, cy = float(pose_m[0, 3]), float(pose_m[1, 3])
            for p in ctrl.env.part_paths:
                if p == part_path:
                    continue
                pp = ctrl.part_world_pose(p)
                if pp is None:
                    continue
                d = float(np.hypot(float(pp[0, 3]) - cx, float(pp[1, 3]) - cy))
                if d < 0.04:
                    n_close += 1
                nearest = d if nearest is None else min(nearest, d)
        scene = {"n_close": n_close,
                 "nearest_mm": None if nearest is None else nearest * 1000.0}
        # always evaluate the exact heuristic, plus N_CANDS perturbations around
        # it — including the tilt / lip-press strategies the scorer trained on
        cands = [{"xy_offset": (0.0, 0.0), "grasp_yaw": h_yaw, "grasp_dz": 0.0,
                  "approach_dh": 0.12, "heuristic_yaw": h_yaw, "width": 1.0,
                  "strategy": "direct"}]
        for _ in range(N_CANDS):
            c = randomize.sample_candidate(rng, kind, heuristic_yaw=h_yaw,
                                           cluttered=n_close > 0)
            c["heuristic_yaw"] = h_yaw
            cands.append(c)
        probs = scorer.score_batch(part, cands, scene=scene)
        best = cands[int(probs.argmax())]
        return best
    return policy


def make_random_policy():
    """Reference policy: execute ONE random perturbed grasp per pick (no scoring).
    Shows what naive random grasping does — the floor the scorer must beat."""
    import random
    from graspsort import randomize
    rng = random.Random(SEED * 90997 + 3)

    def policy(part_path, ctrl):
        kind, _size, _pose = _part_state(ctrl, part_path)
        h_yaw = _heuristic_yaw(ctrl, part_path, kind)
        return randomize.sample_candidate(rng, kind, heuristic_yaw=h_yaw)
    return policy


def run_policy(env, ctrl, name, policy):
    """Run TRIALS sort trials on an ALREADY-BOOTED env (no boot/teardown here —
    Isaac's SimulationApp.close() hard-exits the process; main() closes last)."""
    trials = []
    for t in range(TRIALS):
        rng = random.Random(SEED * 7919 + t)        # SAME scenes across policies
        build_scene(env, ctrl, rng)
        ctrl.reset_to_home()
        res = ctrl.run_sort_trial(policy=policy)
        trials.append({"trial": t, "n_parts": res.n_parts, "n_correct": res.n_correct,
                       "success_rate": res.success_rate, "attempts": res.attempts,
                       "per_part": res.per_part})
        print(f"[eval:{name}] trial {t + 1}/{TRIALS} "
              f"success={res.success_rate:.2f} ({res.n_correct}/{res.n_parts})", flush=True)
    import numpy as np
    rates = [x["success_rate"] for x in trials]
    return {"policy": name, "mean": float(np.mean(rates)) if rates else 0.0,
            "std": float(np.std(rates)) if rates else 0.0, "trials": trials}


def main():
    import tempfile
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort.logging_schema import emit_dir
    stage = tempfile.mkdtemp(prefix="gs_eval_")   # local staging; emit to OUT_DIR (may be s3://)
    report = {"seed": SEED, "trials": TRIALS, "parts_per_trial": PARTS, "results": {}}

    # POLICY: heuristic | scorer | random | both (heur+scorer) | all (heur+random+scorer)
    policies = []
    if POLICY in ("heuristic", "both", "all"):
        policies.append(("heuristic", None))
    if POLICY in ("random", "all"):
        policies.append(("random", make_random_policy()))
    if POLICY in ("scorer", "both", "all"):
        policies.append(("scorer", make_scorer_policy(MODEL)))

    def publish():
        with open(os.path.join(stage, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        emit_dir(stage, OUT_DIR)

    # ONE SimulationApp for all policies; Isaac's close() hard-exits the process,
    # so publish the report BEFORE closing (and after each policy for resilience).
    env = GraspSortEnv(headless=HEADLESS)
    env.reset_world()
    ctrl = GraspSortController(env)
    for name, policy in policies:
        report["results"][name] = run_policy(env, ctrl, name, policy)
        publish()

    if "heuristic" in report["results"] and "scorer" in report["results"]:
        b = report["results"]["heuristic"]["mean"]
        a = report["results"]["scorer"]["mean"]
        report["headline"] = {"before": b, "after": a, "delta": a - b}

    publish()
    print(f"[eval] DONE -> {OUT_DIR}/report.json", flush=True)
    for name, r in report["results"].items():
        print(f"[eval]   {name}: {r['mean']:.3f} +/- {r['std']:.3f}", flush=True)
    env.close()                        # LAST — may hard-exit; report already emitted


if __name__ == "__main__":
    main()
