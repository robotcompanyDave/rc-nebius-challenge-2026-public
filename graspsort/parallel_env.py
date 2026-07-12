"""
Parallel/replicated grasp env — K independent UR10e+gripper+part cells in ONE
physics scene, so one `world.step()` advances all K at once. This is the
"replicate the robot many times in the scene" pattern: the expensive physics
step is shared across K grasp attempts instead of run K times sequentially.

Design (deliberately low-risk): we REUSE the proven single-arm stack verbatim —
`load_arm` builds each cell's arm+gripper, `ArmGripper` is the per-cell I/O, and a
per-cell `GraspSortController` runs the exact validated grasp state machine. The
only new thing here is laying K cells out on a grid and stepping them together.
The single-arm controller already splits control from physics (`ctrl.step(dt)`
does NO physics; the caller steps the world), so the driver just ticks all K
controllers, then calls `env.step()` ONCE.

Cells are spaced far apart on a grid so their parts/arms never interact. Each
cell's controller reads its own arm-root world transform, so the grid offset is
handled automatically by the existing base-frame math.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional

import numpy as np

from .parts import PartSpec, spawn_part
from .sim_env import TABLE_TOP_Z, DEFAULT_USD


CELL_SPACING = 4.0        # metres between grid cells (arms have ~1.3 m reach)


class ParallelGraspEnv:
    def __init__(self, n_envs: int = 16, headless: bool = True,
                 usd_path: Optional[str] = None, table_top_z: float = TABLE_TOP_Z):
        from isaacsim import SimulationApp
        self.sim = SimulationApp({"headless": headless, "width": 640, "height": 480})

        import omni.usd
        import carb
        from isaacsim.core.api import World

        self.n = int(n_envs)
        self.table_top_z = table_top_z
        self.usd_path = usd_path or DEFAULT_USD

        try:
            s = carb.settings.get_settings()
            s.set("/rtx/materialDb/syncLoads", True)
            s.set("/rtx/hydra/materialSyncLoads", True)
        except Exception:
            pass

        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()

        # grid layout: near-square
        self.cols = int(math.ceil(math.sqrt(self.n)))
        self.offsets = []
        for i in range(self.n):
            r, c = divmod(i, self.cols)
            self.offsets.append((c * CELL_SPACING, r * CELL_SPACING))

        self._build_shared_stage()

        self.world = World()
        from .robot import load_arm, ArmGripper
        self.arms: List[ArmGripper] = []
        self.cfgs = []
        for i, (dx, dy) in enumerate(self.offsets):
            arm_path = f"/World/envs/env_{i}/ur10e"
            base = (0.027 + dx, -0.874 + dy, table_top_z)
            cfg = load_arm(self.stage, self.usd_path, arm_path=arm_path, base_pos=base)
            self._build_platform(i, dx, dy)
            arm = ArmGripper(cfg)
            arm.attach(self.world)
            self.arms.append(arm)
            self.cfgs.append(cfg)

        self._next_pid = 0

    # ── stage authoring ──────────────────────────────────────────────────────
    def _build_shared_stage(self):
        from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.Xform.Define(self.stage, "/World")
        UsdGeom.Xform.Define(self.stage, "/World/envs")

        scene = UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(9.81)

        # one big shared ground plane under the whole grid
        span = (self.cols + 1) * CELL_SPACING
        ground = UsdGeom.Cube.Define(self.stage, "/World/Ground")
        ground.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(ground.GetPrim()).SetTranslate(Gf.Vec3d(span / 2, span / 2, -0.5))
        UsdGeom.XformCommonAPI(ground.GetPrim()).SetScale(Gf.Vec3f(span * 2, span * 2, 1.0))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1500.0)

    def _build_platform(self, i, dx, dy):
        from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema
        slab_h = 0.04
        path = f"/World/envs/env_{i}/SortPlatform"
        slab = UsdGeom.Cube.Define(self.stage, path)
        slab.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(slab.GetPrim()).SetTranslate(
            Gf.Vec3d(0.30 + dx, -0.70 + dy, self.table_top_z - slab_h / 2.0))
        UsdGeom.XformCommonAPI(slab.GetPrim()).SetScale(Gf.Vec3f(0.8, 1.2, slab_h))
        slab.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.52)])
        UsdPhysics.CollisionAPI.Apply(slab.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(slab.GetPrim())
        pc.CreateContactOffsetAttr().Set(0.005)
        pc.CreateRestOffsetAttr().Set(0.0)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def reset_world(self):
        self.world.reset()
        for arm in self.arms:
            arm.post_reset()
        for _ in range(8):
            self.world.step(render=False)

    def step(self, render: bool = False):
        self.world.step(render=render)

    def settle(self, ticks: int = 40):
        for _ in range(ticks):
            self.world.step(render=False)

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass

    # ── per-cell parts (POOLED: re-pose instead of respawn) ─────────────────
    # RemovePrim+respawn every round is the suspected driver of the native heap
    # drift after ~100 attempts (HANDOFF §9) — so each cell keeps a pool of
    # spawned parts keyed by (kind, size): matching parts get RE-POSED to the
    # new spec, missing ones spawn once, leftovers PARK (rigid body disabled,
    # translated below the ground plane).
    _PARK_Z = -5.0

    def clear_part(self, i: int):
        """Back-compat single-part clear: park everything active in cell i."""
        self._park_cell(i)

    def _park_cell(self, i: int):
        from .parts import set_part_physics_enabled
        from pxr import UsdGeom, Gf
        view = self._views[i]
        for p in list(view.part_paths):
            set_part_physics_enabled(self.stage, p, False)
            prim = self.stage.GetPrimAtPath(p)
            if prim.IsValid():
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        t = op.Get()
                        op.Set(Gf.Vec3d(t[0], t[1], self._PARK_Z))
                        break
        view.part_paths.clear()
        view.part_specs.clear()
        view.part_kinds.clear()

    def set_cell_parts(self, i: int, specs: List[PartSpec]) -> List[str]:
        """Give cell i exactly `specs` (world-xy already offset to the cell):
        re-pose pooled (kind,size) matches, spawn the rest, park the surplus.
        Returns the active part prim paths, aligned with `specs`."""
        from .parts import repose_part, set_part_physics_enabled
        self._park_cell(i)
        view = self._views[i]
        pool = self._pools[i]
        paths = []
        for spec in specs:
            key = (spec.kind, spec.size)
            bucket = pool.setdefault(key, [])
            free = [p for p in bucket if p not in paths]
            if free:
                path = free[0]
                set_part_physics_enabled(self.stage, path, True)
                repose_part(self.stage, path, spec, self.table_top_z)
            else:
                path = f"/World/envs/env_{i}/NutBolt_{self._next_pid}"
                self._next_pid += 1
                spawn_part(self.stage, path, spec, self.table_top_z)
                bucket.append(path)
            paths.append(path)
            view.part_paths.append(path)
            view.part_specs[path] = spec
            view.part_kinds[path] = spec.kind
        return paths

    def spawn_part_for(self, i: int, spec: PartSpec) -> str:
        """Back-compat single-part API (v1 driver) — pooled underneath."""
        return self.set_cell_parts(i, [spec])[0]

    def cell_view(self, i: int):
        """An env-like shim so a GraspSortController can drive cell i: `.arm`,
        `.table_top_z`, and the per-cell part registry the v2 controller reads
        (`part_paths` / `part_specs` / `part_kinds`; no soft rig yet)."""
        return self._views[i]

    @property
    def _views(self):
        if not hasattr(self, "_views_"):
            self._views_ = [_CellView(self.arms[i], self.table_top_z)
                            for i in range(self.n)]
            self._pools = [{} for _ in range(self.n)]
        return self._views_


class _CellView:
    __slots__ = ("arm", "table_top_z", "part_paths", "part_specs", "part_kinds",
                 "soft_rig")

    def __init__(self, arm, table_top_z):
        self.arm = arm
        self.table_top_z = table_top_z
        self.part_paths: list = []
        self.part_specs: dict = {}
        self.part_kinds: dict = {}
        self.soft_rig = None                     # parallel cells: hard surface (for now)
