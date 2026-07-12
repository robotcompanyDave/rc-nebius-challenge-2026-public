#!/usr/bin/env python3
"""
Press-lift physics probe — STAGE 1 of the one-finger washer pick.

Question (David, 2026-07-02): press ONE open-jaw finger down on one side of a
large flat washer on the soft surface; does the soft-pad physics tilt the
washer so the OPPOSITE edge rises high enough for the other (closing) finger
to get under it? Control the soft-material factors and measure lift — the
grasp itself is stage 2.

Per trial (single Isaac boot, factors mutated live):
  1. set pad params (max_indent / reform_rate / spread_gain; hard stop moved
     to match depth), 2. repose an m12 flat washer at the work centre,
  3. servo one fingertip (gripper OPEN) onto the press point on the near rim,
  4. descend to the commanded press depth with soft.pressing=True and HOLD,
  5. record the far-rim height each tick.

Metrics per trial: peak/end far-rim lift above the nominal surface (mm),
far-rim UNDERSIDE clearance (lift − washer half-thickness), pad max dent,
finger sink. Success tiers: underside above the nominal surface (a finger
denting its own tile to the ik floor gets under) / above +3 mm (gets under
without denting).

    ~/TOOLS/isaac-sim/python.sh tools/probe_press_lift.py
Env: GS_PL_OUT (default data/<date>/<HHMM>-press_lift), GS_PL_HOLD (ticks).
"""
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

_now = datetime.datetime.now()
OUT = os.environ.get("GS_PL_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-press_lift"))
HOLD = int(os.environ.get("GS_PL_HOLD", "150"))
OPENNESS = 0.35            # mostly-open jaw: one finger presses, one hovers wide
HOVER_DH = 0.030           # hover height above the surface before the press
SERVO_TICKS = 240          # settle/servo budget to land the finger on the point
BASE = dict(depth=0.005, reform=0.25, spread=1.5, press=0.005, edge=1.0)
# one-factor-at-a-time around the platform defaults
SWEEP_OFAT = ([("baseline", BASE)]
              + [(f"depth={d*1000:.0f}mm", {**BASE, "depth": d}) for d in (0.004, 0.007, 0.010)]
              + [(f"reform={r}", {**BASE, "reform": r}) for r in (0.10, 0.50)]
              + [(f"spread={s}", {**BASE, "spread": s}) for s in (1.0, 2.5)]
              + [(f"press={p*1000:.0f}mm", {**BASE, "press": p}) for p in (0.003, 0.007, 0.010)]
              + [(f"edge={e}", {**BASE, "edge": e}) for e in (0.7, 0.9, 1.1)])
# refinement grid around the OFAT winner (narrow spread → local dent → pivot):
# spread × edge, plus slow-reform and deeper-stop combos at the best spread
SWEEP_REFINE = ([(f"sp={s}/e={e}", {**BASE, "spread": s, "edge": e})
                 for s in (0.7, 0.9, 1.1) for e in (0.9, 1.0, 1.1)]
                + [("sp=1.0/rf=0.1", {**BASE, "spread": 1.0, "reform": 0.10}),
                   ("sp=1.0/d=7mm", {**BASE, "spread": 1.0, "depth": 0.007,
                                     "press": 0.007}),
                   ("sp=0.9/d=7/rf=.1", {**BASE, "spread": 0.9, "depth": 0.007,
                                         "press": 0.007, "reform": 0.10})])
SWEEP_MINI = [("spread=1.0", {**BASE, "spread": 1.0}),
              ("spread=1.0b", {**BASE, "spread": 1.0})]
SWEEP = {"refine": SWEEP_REFINE, "mini": SWEEP_MINI}.get(
    os.environ.get("GS_PL_SWEEP", ""), SWEEP_OFAT)


def set_soft_params(env, depth, reform, spread):
    """Mutate the live rig: pad field params + move the hard stop to `depth`."""
    from pxr import UsdGeom, Gf
    rig = env.soft_rig
    rig.depth = float(depth)
    rig.pad.max_indent_m = float(depth)
    rig.pad.reform_rate = float(reform)
    rig.pad.spread_gain = float(spread)
    hs = env.stage.GetPrimAtPath("/World/SortHardStop")
    if hs.IsValid():
        top = rig.surface_z - rig.depth
        # HARDSTOP_THICK = 0.02 (soft_rig.py)
        UsdGeom.XformCommonAPI(hs).SetTranslate(
            Gf.Vec3d(rig.centre[0], rig.centre[1], top - 0.01))


def finger_xy_z(stage, prim):
    from pxr import UsdGeom
    bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    b = bc.ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = b.GetMin(), b.GetMax()
    return (0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), float(mn[2]))


def washer_dims(stage, path):
    from pxr import UsdGeom
    bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    b = bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    mn, mx = b.GetMin(), b.GetMax()
    od = max(float(mx[0] - mn[0]), float(mx[1] - mn[1]))
    thick = float(mx[2] - mn[2])
    return od, thick


def far_rim_z(ctrl, path, u, od):
    """World z of the rim point diametrically opposite the press direction u."""
    pose = ctrl.part_world_pose(path)
    if pose is None:
        return None
    R, t = pose[:3, :3], pose[:3, 3]
    d = R.T @ np.array([-u[0], -u[1], 0.0])
    d[2] = 0.0                                   # stay in the disc plane
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    p = t + R @ (d / n * od / 2.0)
    return float(p[2])


def main():
    from graspsort.sim_env import GraspSortEnv
    from graspsort.controller import GraspSortController, _GRASP_R_WORLD
    from graspsort import parts

    env = GraspSortEnv(headless=True)
    env.reset_world()
    ctrl = GraspSortController(env)
    rig = env.soft_rig
    assert rig is not None, "probe needs GS_SOFT=1"
    surface = rig.surface_z
    centre = rig.centre
    dt = ctrl.dt

    u = np.array([1.0, 0.0])                     # press direction: world +X
    yaw = 0.0                                    # jaw axis along u
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ _GRASP_R_WORLD

    spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                          xy=centre, rotz_deg=0.0)
    path = env.spawn_parts([spec])[0]
    env.settle(60)
    od, thick = washer_dims(env.stage, path)
    print(f"[press] m12 washer OD={od*1000:.1f}mm thick={thick*1000:.1f}mm "
          f"surface_z={surface:.4f}", flush=True)

    def tickn(n, state="descend_pick"):
        ctrl._state = state
        for _ in range(n):
            ctrl._auto_drive(dt)
            env.step(render=False)

    def drive_to(world_pos, ticks, state="descend_pick"):
        ctrl._set_goal_world(np.asarray(world_pos, dtype=float), R)
        tickn(ticks, state)

    os.makedirs(OUT, exist_ok=True)
    results = []
    fjson = open(os.path.join(OUT, "trials.jsonl"), "w")

    for label, f in SWEEP:
        set_soft_params(env, f["depth"], f["reform"], f["spread"])
        from graspsort.parts import repose_part
        rig.press_foot = None
        rig.pressing = False
        env.arm.set_gripper(OPENNESS)
        # home-ish hover away from the washer so reform + repose are clean
        drive_to((centre[0], centre[1], surface + 0.20), 90, state="lift")
        repose_part(env.stage, path, spec, env.table_top_z)
        env.settle(50)

        press_xy = (centre[0] + u[0] * od / 2.0 * f["edge"],
                    centre[1] + u[1] * od / 2.0 * f["edge"])
        fingers = [p for p in rig._finger_prims if p and p.IsValid()]

        # hover, then servo the NEAR finger onto the press point
        drive_to((press_xy[0], press_xy[1], surface + HOVER_DH), 120)
        goal = np.array([press_xy[0], press_xy[1], surface + HOVER_DH])
        for _ in range(4):
            fa = min(fingers, key=lambda p: (finger_xy_z(env.stage, p)[0] - press_xy[0]) ** 2
                                            + (finger_xy_z(env.stage, p)[1] - press_xy[1]) ** 2)
            fx, fy, _ = finger_xy_z(env.stage, fa)
            goal[0] += press_xy[0] - fx
            goal[1] += press_xy[1] - fy
            drive_to(goal, SERVO_TICKS // 4)
        fx, fy, fz0 = finger_xy_z(env.stage, fa)
        miss0 = math.hypot(fx - press_xy[0], fy - press_xy[1])

        # press: descend the commanded depth below the surface and HOLD
        rig.pressing = True
        goal[2] = surface - f["press"]
        ctrl._set_goal_world(goal, R)
        lift_mm, dent_mm, sink_mm = [], 0.0, 0.0
        ctrl._state = "descend_pick"
        for k in range(HOLD):
            ctrl._auto_drive(dt)
            env.step(render=False)
            z = far_rim_z(ctrl, path, u, od)
            if z is not None:
                lift_mm.append((z - surface) * 1000.0)
            dent_mm = max(dent_mm, rig.max_dent_mm())
            sink_mm = max(sink_mm, (surface - finger_xy_z(env.stage, fa)[2]) * 1000.0)
        rig.pressing = False

        peak = max(lift_mm) if lift_mm else float("nan")
        end = lift_mm[-1] if lift_mm else float("nan")
        # SUSTAINED lift = median of the last 30 hold ticks — several combos
        # show a large transient pop that does not hold; the grasp needs a lip
        # that is still there when the closing finger arrives.
        tail = sorted(lift_mm[-30:])
        sustained = tail[len(tail) // 2] if tail else float("nan")
        under_sus = sustained - thick * 500.0    # rim centre → underside (≈ −t/2)
        rec = {"label": label, **{k: round(v, 4) for k, v in f.items()},
               "press_miss_mm": round(miss0 * 1000, 1),
               "peak_lift_mm": round(peak, 2), "end_lift_mm": round(end, 2),
               "sustained_lift_mm": round(sustained, 2),
               "underside_sustained_mm": round(under_sus, 2),
               "ok_at_surface": bool(under_sus > 0.0),
               "ok_no_dent": bool(under_sus > 3.0),
               "pad_dent_mm": round(dent_mm, 2), "finger_sink_mm": round(sink_mm, 2),
               "curve_mm": [round(v, 2) for v in lift_mm[::5]]}
        results.append(rec)
        fjson.write(json.dumps(rec) + "\n")
        fjson.flush()
        print(f"[press] {label:14s} peak={peak:6.2f}mm end={end:6.2f}mm "
              f"sustained={sustained:6.2f}mm underside={under_sus:6.2f}mm dent={dent_mm:.1f}mm "
              f"sink={sink_mm:.1f}mm miss={miss0*1000:.1f}mm "
              f"{'OK' if under_sus > 0 else '--'}", flush=True)

    fjson.close()
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump({"washer_od_mm": od * 1000, "washer_thick_mm": thick * 1000,
                   "hold_ticks": HOLD, "openness": OPENNESS,
                   "results": results}, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in results:
            ax.plot(np.arange(len(r["curve_mm"])) * 5, r["curve_mm"],
                    label=f"{r['label']} (peak {r['peak_lift_mm']:.1f})")
        ax.axhline(thick * 500.0, color="k", ls="--", lw=1,
                   label="underside at surface")
        ax.set_xlabel("hold ticks"); ax.set_ylabel("far-rim lift above surface (mm)")
        ax.set_title("Press-lift: far-rim height while pressing the near rim")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "lift_curves.png"), dpi=130)
        print(f"[press] wrote {OUT}/lift_curves.png", flush=True)
    except Exception as e:
        print(f"[press] plot skipped: {e}", flush=True)

    print(f"[press] DONE {len(results)} trials -> {OUT}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
