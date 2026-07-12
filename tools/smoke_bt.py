#!/usr/bin/env python3
"""
Smoke the behaviour-tree sorter (graspsort/behavior.py) headless: mixed
nut/bolt/washer scene INCLUDING the M6-washer-flat training target, sort with
strategy escalation, dump the BT trace + SortResult.

    ~/TOOLS/isaac-sim/python.sh tools/smoke_bt.py
Env: GS_BT_OUT (default data/<today>/<now>-bt_smoke), GS_MODEL (scorer .npz,
optional — no model → heuristic within the ladder), GS_BT_SEED.
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = int(os.environ.get("GS_BT_SEED", "7"))
_now = datetime.datetime.now()
OUT = os.environ.get("GS_BT_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-bt_smoke"))
MODEL = os.environ.get("GS_MODEL", "")


def main():
    import random
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController
    from graspsort.behavior import SortBT

    env = GraspSortEnv(headless=True)
    env.reset_world()
    ctrl = GraspSortController(env)

    from graspsort import parts
    rng = random.Random(SEED)
    centre = ctrl.work_centre_xy()
    scen = [("nut", "flat", "m12"), ("bolt", "on-side", "m12"),
            ("washer", "flat", "m12"), ("washer", "flat", "m6")]
    xys = parts.scatter_xy(centre, len(scen), spread=0.06, min_sep=0.05, rng=rng)
    specs = [parts.PartSpec(kind=k, size=sz, pose=p, xy=xys[i],
                            rotz_deg=rng.uniform(0, 360))
             for i, (k, p, sz) in enumerate(scen)]
    env.spawn_parts(specs)
    env.settle(60)

    scorer = None
    if MODEL and os.path.isfile(MODEL):
        from graspsort.scorer import GraspScorer
        scorer = GraspScorer.load(MODEL)
        print(f"[bt-smoke] scorer: {MODEL} (val AUC={scorer.meta.get('val_auc')})",
              flush=True)
    else:
        print("[bt-smoke] no scorer — heuristic within the strategy ladder", flush=True)

    bt = SortBT(env, ctrl, scorer=scorer, seed=SEED)
    res = bt.run()

    os.makedirs(OUT, exist_ok=True)
    bt.save_trace(os.path.join(OUT, "trace.jsonl"))
    summary = {"n_parts": res.n_parts, "n_correct": res.n_correct,
               "attempts": res.attempts, "per_part": res.per_part,
               "escalations": sum(1 for r in bt.bb.trace
                                  if r["event"] == "enter"
                                  and r["node"] == "propose_grasp"
                                  and r.get("strategy") != "direct")}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[bt-smoke] sorted {res.n_correct}/{res.n_parts} in {res.attempts} picks "
          f"({summary['escalations']} escalated attempts) -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
