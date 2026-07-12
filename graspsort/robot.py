"""
UR10e articulation interface — load the arm, set up the gripper, and expose the
low-level joint/gripper I/O the controller drives.

Ports the load-bearing robot-init from the private platform
(`adapter.py::post_reset`, 528–617): discover the 6 arm DOF indices by joint
name, FORCE the arm to HOME (USD initial-state authoring is unreliable), set high
drive gains (kps 1e5 / kds 1e3), and assert the HOME targets immediately.

Two gripper rungs, selected by GS_GRIPPER (default "parametric"):
  parametric — the straight-line parallel jaw (two prismatic DOFs, gripper.py);
               the WORKING data-gen rung. Pads run to the TCP (functionally the
               platform's fingernail tips), and the scorer features are frame-
               invariant (yaw DELTAS) so policies port to the real gripper.
  robotiq    — the REAL Robotiq 2F-85 baked into the platform arm USD. Wired
               (fingernail tips, friction/FMAX, closing-arc model, platform-
               scene composition) but currently BLOCKED in this standalone
               harness: the five-bar mimic linkage locks at ~1° when composed
               outside the live gateway (probed exhaustively — drive stiffness,
               tips off, PhysicsScene parity, GPU pipeline, self-collisions,
               solver iterations, gearing flips, and the platform's own scene
               file all reproduce the lock; see tools/probe_gripper.py). The
               same USD demonstrably closes on the live platform, so the gap is
               harness-level PhysX behaviour, not the asset. Revisit with an
               Isaac-version bump or NVIDIA's mimic-joint guidance.

All Isaac imports are local to methods so this module is import-safe before a
SimulationApp exists (only `kinematics` is touched at import time).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import kinematics
from .kinematics import JOINT_NAMES, NUM_JOINTS, HOME_POSITION
from .gripper import GripperParams, build_gripper
from .gripper_robotiq import RobotiqParams, setup_robotiq


def find_link(stage, suffix: str) -> Optional[str]:
    """First prim path whose name ends with `suffix` (e.g. 'wrist_3_link')."""
    from pxr import Usd
    for prim in stage.Traverse():
        if prim.GetName().endswith(suffix):
            return prim.GetPath().pathString
    return None


def load_arm(stage, usd_path: str, arm_path: str = "/World/ur10e",
             base_pos=(0.027, -0.874, 1.028), base_rot_z_deg: float = 0.0,
             gripper: Optional[str] = None) -> dict:
    """Reference the UR10e USD into the stage and set up the selected gripper.

    Default base pose matches the kinematics HOME derivation (kinematics.py:75):
    base upright at world ≈ (0.027, −0.874, 1.028) facing +X → at HOME the tool
    points dead-down over the work zone, ~13 cm above the table. Returns config
    consumed by `ArmGripper`."""
    from pxr import UsdGeom, Gf, Sdf
    gripper = (gripper or os.environ.get("GS_GRIPPER", "parametric")).strip().lower()
    prim = stage.DefinePrim(arm_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetTranslate(Gf.Vec3d(*base_pos))
    if base_rot_z_deg:
        xf.SetRotate(Gf.Vec3f(0.0, 0.0, base_rot_z_deg))

    tool_link = find_link(stage, "wrist_3_link")
    if tool_link is None:
        raise RuntimeError(f"wrist_3_link not found under {arm_path}; "
                           f"check the UR USD ({usd_path})")
    if gripper == "robotiq":
        gconf = setup_robotiq(stage, tool_link, RobotiqParams())
    else:
        # Parametric rung: DEACTIVATE the baked-in Robotiq (if this arm USD has
        # one) so the two grippers never compose together, then author the
        # parallel jaw at its own mount path.
        for sub in ("/gripper", "/gripper_fix"):
            gp = stage.GetPrimAtPath(tool_link + sub)
            if gp and gp.IsValid():
                gp.SetActive(False)
        gconf = build_gripper(stage, tool_link, GripperParams())
    return dict(arm_path=arm_path, tool_link=tool_link, gripper=gconf)


def platform_arm_cfg(stage, arm_path: str = "/World/big_table/ur10e") -> dict:
    """Use the UR10e+2F-85 ALREADY COMPOSED in the platform scene (the slim ur10e
    site overlay) — no referencing, no mounting: the exact rig the live platform
    grasps with. Applies the fingernail tips + friction/FMAX on top, same as the
    platform adapter does at post_reset."""
    prim = stage.GetPrimAtPath(arm_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"platform arm not found at {arm_path}")
    tool_link = None
    from pxr import Usd
    for d in Usd.PrimRange(prim):
        if d.GetName().endswith("wrist_3_link"):
            tool_link = d.GetPath().pathString
            break
    if tool_link is None:
        raise RuntimeError(f"wrist_3_link not found under {arm_path}")
    gconf = setup_robotiq(stage, tool_link, RobotiqParams())
    return dict(arm_path=arm_path, tool_link=tool_link, gripper=gconf)


class ArmGripper:
    """Wraps the Isaac SingleArticulation + the kinematics controller for one arm."""

    def __init__(self, robot_cfg: dict):
        self.cfg = robot_cfg
        self.tool_link = robot_cfg["tool_link"]
        self.gconf = robot_cfg["gripper"]
        self.tool_offset = np.array([0.0, float(self.gconf["reach"]), 0.0])
        self.controller = kinematics.UR10Controller(tool_offset=self.tool_offset)

        self._robot = None
        self._arm_idx: Optional[list] = None
        self._grip_idx: Optional[list] = None      # drive-joint DOF indices (1 robotiq / 2 parametric)
        self._targets: Optional[np.ndarray] = None
        self.gripper_openness = 0.0                 # 0 = open, 1 = closed (UR convention)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def attach(self, world=None):
        """Create the SingleArticulation and register it with the World scene.

        Registration (`world.scene.add`) is load-bearing: it is what makes
        `world.reset()` INITIALIZE the articulation (populate the physics view /
        `dof_names`). Without it `dof_names` is None and `post_reset` blows up
        ('Articulation needs to be initialized'). Mirrors adapter.py:1152."""
        from isaacsim.core.prims import SingleArticulation
        art = SingleArticulation(prim_path=self.cfg["arm_path"], name="ur10e")
        self._robot = world.scene.add(art) if world is not None else art
        return self._robot

    def post_reset(self):
        """Discover DOFs, force HOME, set drive gains, assert targets (adapter.py:528)."""
        from isaacsim.core.utils.types import ArticulationAction
        r = self._robot
        if r is None:
            return
        dof_names = list(r.dof_names)

        self._arm_idx = []
        for jname in JOINT_NAMES:
            for i, dn in enumerate(dof_names):
                if jname in dn:
                    self._arm_idx.append(i)
                    break

        full_q = r.get_joint_positions().copy()
        if len(self._arm_idx) == NUM_JOINTS:
            home = HOME_POSITION.copy()
            self.controller.joint_positions = home
            for i, idx in enumerate(self._arm_idx):
                full_q[idx] = home[i]
            r.set_joint_positions(full_q)
        self._targets = full_q.copy()

        self._grip_idx = []
        for jname in self.gconf["drive_joints"]:
            for i, dn in enumerate(dof_names):
                if jname in dn:
                    self._grip_idx.append(i)
                    break

        # ── DIAGNOSTIC (Step 0, GS_DEBUG): are the fingers real articulation DOFs? ──
        if os.environ.get("GS_DEBUG", "0") not in ("0", "", "false", "False"):
            print(f"[robot] num_dof={getattr(r, 'num_dof', '?')} dof_names={dof_names}", flush=True)
            print(f"[robot] arm_idx={self._arm_idx} (need {NUM_JOINTS}); "
                  f"grip_idx={self._grip_idx} looking_for="
                  f"{self.gconf['drive_joints']!r}", flush=True)

        try:
            n = r.num_dof
            r.get_articulation_controller().set_gains(
                kps=np.full(n, 1e5), kds=np.full(n, 1e3))
        except Exception:
            pass
        self.set_gripper(0.0)        # open
        r.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=self._targets.copy()))

    # ── arm ─────────────────────────────────────────────────────────────────
    def apply_arm(self, q: np.ndarray):
        from isaacsim.core.utils.types import ArticulationAction
        if self._robot is None or self._arm_idx is None or self._targets is None:
            return
        if len(self._arm_idx) != NUM_JOINTS:
            return
        for i, idx in enumerate(self._arm_idx):
            self._targets[idx] = q[i]
        self._robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=self._targets.copy()))

    def go_home(self):
        q = self.controller.go_home()
        self.apply_arm(q)
        self.set_gripper(0.0)

    # ── gripper ───────────────────────────────────────────────────────────────
    def set_gripper(self, openness: float):
        """openness 0=open → 1=closed. Drives both prismatic finger DOFs."""
        from isaacsim.core.utils.types import ArticulationAction
        self.gripper_openness = float(np.clip(openness, 0.0, 1.0))
        if self._robot is None or not self._grip_idx or self._targets is None:
            return
        ov, cv = self.gconf["open_val"], self.gconf["close_val"]
        pos = ov + self.gripper_openness * (cv - ov)
        for idx in self._grip_idx:
            self._targets[idx] = pos
        self._robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=self._targets.copy()))

    def gripper_actual_openness(self) -> Optional[float]:
        """ACTUAL finger position as openness [0..1]; the gap vs commanded = stall
        (adapter.py:2069). Averages the two pads."""
        if self._robot is None or not self._grip_idx:
            return None
        try:
            jp = self._robot.get_joint_positions()
            ov, cv = self.gconf["open_val"], self.gconf["close_val"]
            if abs(cv - ov) < 1e-9:
                return None
            vals = [(float(jp[idx]) - ov) / (cv - ov) for idx in self._grip_idx]
            return float(np.mean(vals))
        except Exception:
            return None

    def gripper_effort(self) -> Optional[float]:
        """Measured effort on the finger drives (force-feedback grasp sensing,
        adapter.py:1883). Max magnitude across the two pads."""
        if self._robot is None or not self._grip_idx:
            return None
        try:
            eff = self._robot.get_measured_joint_efforts()
            if eff is None:
                return None
            return float(max(abs(float(eff[idx])) for idx in self._grip_idx
                             if idx < len(eff)))
        except Exception:
            return None

    # ── state ─────────────────────────────────────────────────────────────────
    def arm_root_world_matrix(self) -> Optional[np.ndarray]:
        """4×4 world transform of the arm base (for world↔base goal conversion)."""
        from pxr import Usd, UsdGeom
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.cfg["arm_path"])
            m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return np.array([[m[i][j] for j in range(4)] for i in range(4)]).T
        except Exception:
            return None
