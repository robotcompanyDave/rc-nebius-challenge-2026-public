"""
Headless grasp+sort simulation environment.

A minimal in-code Isaac stage: physics scene + ground + lights + a static work
platform + a UR10e arm with the parametric gripper. No workshop digital twin, no
TCP gateway, no teleop — just the pieces a grasp/sort rollout needs. Replicates
only the boot + step primitives from the platform gateway (gateway.py:322–357,
1342–1381) and drops everything else.

Boot order is load-bearing: `SimulationApp(...)` MUST be constructed before any
`pxr`/`omni` import or the Omniverse python bindings fail to load. So this module
imports those lazily, inside `GraspSortEnv.__init__`, after the app exists.

Usage:
    env = GraspSortEnv(headless=True)
    env.reset_world()
    paths = env.spawn_parts([...])
    for _ in range(60): env.step()
    ...
    env.close()
"""
from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np

from .parts import PartSpec, spawn_part, scatter_xy

# Work surface height. Kept at the platform's table height so the kinematics HOME
# pose (tool ~13 cm above the table, well-conditioned) holds without retuning.
TABLE_TOP_Z = 1.028
_ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
# Prefer the gripper-baked arm copy (real Robotiq rung; see assets/README.md);
# fall back to the plain arm USD (parametric rung). GS_ARM_USD overrides.
_BAKED_USD = os.path.join(_ASSETS, "virtual", "imports", "ur10", "ur10.usd")
DEFAULT_USD = os.environ.get("GS_ARM_USD") or (
    _BAKED_USD if os.path.isfile(_BAKED_USD) else os.path.join(_ASSETS, "ur10.usd"))
# The PLATFORM scene (preferred): the slim ur10e site overlay copied verbatim from
# rc-remote-platform (assets/platform mirrors the repo layout so its relative
# references resolve). It composes the WORKING UR10e + Robotiq 2F-85 + sort
# platform — the exact rig the live platform grasps with. Raw re-composition of
# the 2F-85 in a from-scratch stage left the five-bar linkage locked at ~1°
# (mimic constraints fought the drive; see gripper_robotiq.py notes), so using
# the platform's own composition is both the fix and the higher-fidelity rung.
PLATFORM_SCENE = os.environ.get("GS_PLATFORM_SCENE") or os.path.join(
    _ASSETS, "platform", "sites", "ur10e", "virtual", "scene.usda")
PLATFORM_ARM = "/World/big_table/ur10e"
PLATFORM_RISER = 0.02          # SortPlatform top sits this far above the table


class GraspSortEnv:
    def __init__(self, headless: bool = True, width: int = 1280, height: int = 720,
                 usd_path: Optional[str] = None, table_top_z: float = TABLE_TOP_Z):
        from isaacsim import SimulationApp
        self.sim = SimulationApp({"headless": headless, "width": width, "height": height})

        # safe to import the Omniverse stack now
        import omni.usd
        import carb
        from isaacsim.core.api import World

        self.table_top_z = table_top_z
        self.usd_path = usd_path or DEFAULT_USD

        # RTX texture-streaming caps — cheap insurance on a heavy scene (gateway.py:335)
        try:
            settings = carb.settings.get_settings()
            settings.set("/rtx/materialDb/syncLoads", True)
            settings.set("/rtx/hydra/materialSyncLoads", True)
        except Exception:
            pass

        # scene source: the in-code stage (default; the parametric-jaw data-gen
        # rung) or the platform slim-site scene (GS_SCENE=platform — the robotiq
        # experiment rung; see robot.py on why robotiq is currently blocked).
        self.platform_scene = (os.environ.get("GS_SCENE", "builtin") == "platform"
                               and os.path.isfile(PLATFORM_SCENE))
        if self.platform_scene:
            omni.usd.get_context().open_stage(PLATFORM_SCENE)
            self.stage = omni.usd.get_context().get_stage()
            self.table_top_z = table_top_z + PLATFORM_RISER   # the sort surface
            print(f"[env] platform scene: {PLATFORM_SCENE} "
                  f"(surface z={self.table_top_z})", flush=True)
        else:
            omni.usd.get_context().new_stage()
            self.stage = omni.usd.get_context().get_stage()
            self._build_stage()

        self.world = World()
        # Optional PhysX GPU dynamics (off = World defaults, matching the gateway)
        if os.environ.get("GS_GPU_PHYS", "0") not in ("0", "", "false"):
            try:
                pc = self.world.get_physics_context()
                pc.enable_gpu_dynamics(True)
                pc.set_broadphase_type("GPU")
                print("[env] PhysX GPU dynamics ON", flush=True)
            except Exception as e:
                print(f"[env] GPU dynamics unavailable: {e}", flush=True)
        # import here (post-app) to keep module import-safe
        from .robot import load_arm, platform_arm_cfg, ArmGripper
        if self.platform_scene:
            self.robot_cfg = platform_arm_cfg(self.stage, PLATFORM_ARM)
        else:
            self.robot_cfg = load_arm(self.stage, self.usd_path,
                                      base_pos=(0.027, -0.874, table_top_z))
        self.arm = ArmGripper(self.robot_cfg)
        self.arm.attach(self.world)

        self._part_paths: List[str] = []
        self._part_kinds: dict = {}
        self._part_specs: dict = {}          # path -> PartSpec (kind/size/pose for the scorer)
        self._next_pid = 0
        self._reset_done = False

        # ── compliant work surface (GS_SOFT, default ON): tiles + 5 mm hard stop +
        # gripper collision filter. Parts spawned later are filtered onto the tiles.
        from .soft_rig import SOFT_ON, SoftRig
        self.soft_rig = None
        if SOFT_ON:
            from .gripper_robotiq import finger_prims, gripper_body_prims
            tool_link = self.robot_cfg["tool_link"]
            fps = finger_prims(self.stage, tool_link)
            if not fps:
                # parametric rung: the pads are the fingers
                gp = self.robot_cfg["gripper"].get("gripper_path", "")
                fps = [self.stage.GetPrimAtPath(gp + "/finger_left"),
                       self.stage.GetPrimAtPath(gp + "/finger_right")]
                fps = [p for p in fps if p and p.IsValid()]
            gbs = gripper_body_prims(self.stage, tool_link) or fps
            # pick-zone centre = FK tool tip at HOME, through the arm's ACTUAL
            # world transform (matches the controller's work anchor)
            centre = self._work_centre_from_stage()
            sinks = (["/World/big_table/SortPlatform", "/World/big_table/TableCollision"]
                     if self.platform_scene else ["/World/SortPlatform"])
            self.soft_rig = SoftRig(self.stage, centre, surface_z=self.table_top_z,
                                    platform_paths=sinks).build(gbs, fps)
            self.soft_rig.set_parts_source(lambda: list(self._part_paths))

    def _work_centre_from_stage(self):
        """World XY of the HOME tool tip: arm root world transform ∘ HOME FK."""
        from pxr import Usd, UsdGeom
        from .kinematics import forward_kinematics_full, HOME_POSITION
        T = forward_kinematics_full(HOME_POSITION, tool_offset=self.arm.tool_offset)
        prim = self.stage.GetPrimAtPath(self.robot_cfg["arm_path"])
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        M = np.array([[m[i][j] for j in range(4)] for i in range(4)]).T
        w = M @ np.append(T[:3, 3], 1.0)
        return (float(w[0]), float(w[1]))

    # ── stage authoring ──────────────────────────────────────────────────────
    def _build_stage(self):
        from pxr import UsdGeom, UsdPhysics, UsdLux, Gf, Sdf, PhysxSchema

        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.Xform.Define(self.stage, "/World")

        # NO authored PhysicsScene: the platform's workshop scene has none either —
        # isaacsim World() creates its default scene (TGS + PhysX defaults), and the
        # Robotiq 2F-85 MIMIC linkage only closes under those defaults. A bare
        # UsdPhysics.Scene authored here LOCKED the linkage at ~1° (measured:
        # commanded full close, finger_joint stalled at 0.016 rad, drive effort
        # ~0.025 N·m regardless of drive stiffness/tips).

        # ground plane at z=0 (collision)
        ground = UsdGeom.Cube.Define(self.stage, "/World/Ground")
        ground.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(ground.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, -0.5))
        UsdGeom.XformCommonAPI(ground.GetPrim()).SetScale(Gf.Vec3f(10.0, 10.0, 1.0))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        # static work platform: a thin slab whose TOP sits at table_top_z
        slab_h = 0.04
        slab = UsdGeom.Cube.Define(self.stage, "/World/SortPlatform")
        slab.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(slab.GetPrim()).SetTranslate(
            Gf.Vec3d(0.30, -0.70, self.table_top_z - slab_h / 2.0))
        UsdGeom.XformCommonAPI(slab.GetPrim()).SetScale(Gf.Vec3f(0.8, 1.2, slab_h))
        slab.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.52)])
        UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(slab.GetPrim())
        pc.CreateContactOffsetAttr().Set(0.005)
        pc.CreateRestOffsetAttr().Set(0.0)

        # lights
        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1500.0)
        key = UsdLux.DistantLight.Define(self.stage, "/World/KeyLight")
        key.CreateIntensityAttr(2500.0)
        UsdGeom.XformCommonAPI(key.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 10.0, 0.0))

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def reset_world(self):
        self.world.reset()
        self.arm.post_reset()
        # warm-up steps so cameras/physics settle
        for _ in range(8):
            self.world.step(render=False)
        self._reset_done = True

    def step(self, render: bool = False):
        if self.soft_rig is not None:
            self.soft_rig.step()        # drive the tile heightfield before physics
        self.world.step(render=render)

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass

    # ── parts ──────────────────────────────────────────────────────────────────
    def clear_parts(self):
        for p in self._part_paths:
            if self.stage.GetPrimAtPath(p).IsValid():
                self.stage.RemovePrim(p)
        self._part_paths.clear()
        self._part_kinds.clear()
        self._part_specs.clear()

    def spawn_parts(self, specs: List[PartSpec]) -> List[str]:
        out = []
        for spec in specs:
            path = f"/World/NutBolt_{self._next_pid}"
            self._next_pid += 1
            spawn_part(self.stage, path, spec, self.table_top_z)
            if self.soft_rig is not None:
                self.soft_rig.filter_part(path)   # the part rides the tiles
            self._part_paths.append(path)
            self._part_kinds[path] = spec.kind
            self._part_specs[path] = spec
            out.append(path)
        return out

    def settle(self, ticks: int = 40):
        for _ in range(ticks):
            self.world.step(render=False)

    @property
    def part_paths(self) -> List[str]:
        return list(self._part_paths)

    @property
    def part_kinds(self) -> dict:
        return dict(self._part_kinds)

    @property
    def part_specs(self) -> dict:
        return dict(self._part_specs)
