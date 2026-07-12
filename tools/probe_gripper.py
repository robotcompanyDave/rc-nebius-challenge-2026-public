#!/usr/bin/env python3
"""Probe the Robotiq finger_joint: command full close, watch the ACTUAL joint
track (or not), and dump the drive parameters PhysX actually has. Diagnoses the
close-lag seen in the v2 smoke (clamp fired at cmd=0.40 with actual~0.04)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graspsort.sim_env import GraspSortEnv


def main():
    env = GraspSortEnv(headless=True)
    env.reset_world()
    arm = env.arm
    r = arm._robot
    print("[probe] dof_names:", list(r.dof_names), flush=True)
    print("[probe] grip_idx:", arm._grip_idx, flush=True)

    # dump ALL gripper revolute joints: limits, drives, mimic API presence
    from pxr import Usd, UsdPhysics
    stage = env.stage
    tool = env.robot_cfg["tool_link"]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(tool)):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
            ax = j.GetAxisAttr().Get()
            d = UsdPhysics.DriveAPI.Get(prim, "angular")
            ds = (f"stiff={d.GetStiffnessAttr().Get()} damp={d.GetDampingAttr().Get()} "
                  f"maxF={d.GetMaxForceAttr().Get()}") if d else "no-drive"
            mim = [s for s in prim.GetAppliedSchemas() if "Mimic" in s]
            print(f"[probe] {prim.GetName()}: axis={ax} limits=[{lo},{hi}] {ds} "
                  f"mimic={mim} enabled={prim.IsActive()}", flush=True)

    # articulation-level gains
    try:
        ac = r.get_articulation_controller()
        gains = ac.get_gains()
        print("[probe] articulation gains kps:", gains[0], flush=True)
        print("[probe] articulation gains kds:", gains[1], flush=True)
    except Exception as e:
        print("[probe] get_gains failed:", e, flush=True)

    # command CLOSED and watch ALL gripper joints track
    gnames = [dn for dn in r.dof_names if "finger" in dn or "knuckle" in dn]
    gidx = [i for i, dn in enumerate(r.dof_names) if dn in gnames]
    arm.set_gripper(1.0)
    for t in range(180):
        env.step(render=False)
        if t % 30 == 0:
            jp = r.get_joint_positions()
            vals = " ".join(f"{dn}={float(jp[i]):+.4f}" for dn, i in zip(gnames, gidx))
            print(f"[probe] t={t:3d} {vals}", flush=True)

    # UNITS TEST: raw target 30. Degrees-drive → joint lands ~0.5236 rad (30°).
    # Radians-drive with a [0, 0.8203 rad] limit → clamps near 0.8203. Rejected →
    # stays ~0.016. Also verify PHYSICAL motion via the fingertip world bbox.
    import numpy as np
    from pxr import UsdGeom
    from isaacsim.core.utils.types import ArticulationAction

    def tip_z():
        bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        zs = []
        for pr in Usd.PrimRange(stage.GetPrimAtPath(tool)):
            if pr.GetName() in ("left_inner_finger", "right_inner_finger"):
                wb = bc.ComputeWorldBound(pr).ComputeAlignedBox()
                zs.append(float(wb.GetMin()[2]))
        return min(zs) if zs else float("nan")

    z0 = tip_z()
    arm._targets[arm._grip_idx[0]] = 30.0
    r.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=arm._targets.copy()))
    for t in range(240):
        env.step(render=False)
        if t % 40 == 0:
            jp = r.get_joint_positions()
            print(f"[probe:30] t={t:3d} raw_tgt=30 "
                  f"joint_rad={float(jp[arm._grip_idx[0]]):.4f} "
                  f"tip_drop_mm={(z0 - tip_z()) * 1000:.1f}", flush=True)
    print("[probe] DONE", flush=True)
    env.close()


if __name__ == "__main__":
    main()
