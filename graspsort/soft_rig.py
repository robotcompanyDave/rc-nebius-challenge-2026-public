"""
Compliant work-surface rig — the platform's poke-v3 tile coupling, slimmed for
the headless training env (port of adapter.py::_ensure_soft_surface /
_build_hard_stop / _build_soft_tiles / _drive_soft_tiles / _soft_feet on
feat/ur10e-soft-sort; visual skin omitted — this rig is physics only).

What it builds (all gated on GS_SOFT, default ON):
  • collision FILTER: every gripper rigid-body link is filtered from the
    platform slab, so the fingertips pass the nominal surface;
  • a rigid HARD STOP whose top face sits `depth` (default 5 mm — the max
    soft-surface depth) below the surface: the physical backstop, NOT filtered;
  • a grid of invisible KINEMATIC TILES over the pick zone that parts rest on
    and RIDE: each step the tile tops are driven to the analytic SoftPad height
    field (dent under the fingertip feet, cosine falloff, eased reform), with
    low min-combine friction so a part on a tilted tile slides into the valley;
  • an IK floor `ik_depth` (default 3 mm) the planner may descend to — kept
    under the hard stop so a plan can never reach it.

Part-press coupling (the press-for-lip strategy): while `pressing` is set, a
fingertip overlapping a part's top ALSO dents the pad at the tip's XY by the
overlap + the part's current sink — the part is rigid, so pushing it down is
depth the pad must yield. A `press_foot` phantom holds the dent shape (memory
foam) while the gripper hops and re-approaches the raised lip.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .softsurface import SoftPad

SOFT_ON = os.environ.get("GS_SOFT", "1") not in ("0", "", "false", "False")
SOFT_DEPTH = float(os.environ.get("GS_SOFT_DEPTH", "0.005"))    # hard stop (max depth)
SOFT_IK = float(os.environ.get("GS_SOFT_IK", "0.003"))          # planner floor
TILE_SPAN = float(os.environ.get("GS_TILE_SPAN", "0.24"))
TILE_CELL = float(os.environ.get("GS_TILE_CELL", "0.012"))
TILE_HALF = 0.004
TILE_FRICTION = 0.01
FOOT_HW = 0.008
REFORM = float(os.environ.get("GS_SOFT_REFORM", "0.25"))
SPREAD_GAIN = float(os.environ.get("GS_SOFT_SPREAD", "1.5"))
HARDSTOP_THICK = 0.02


class SoftRig:
    def __init__(self, stage, work_centre_xy, surface_z: float,
                 platform_paths=None,
                 depth: float = SOFT_DEPTH, ik_depth: float = SOFT_IK):
        self.stage = stage
        self.surface_z = float(surface_z)
        self.depth = float(depth)
        self.ik_depth = float(ik_depth)
        self.platform_paths = list(platform_paths or ["/World/SortPlatform"])
        self.centre = (float(work_centre_xy[0]), float(work_centre_xy[1]))
        self.pad = SoftPad(spread_base_m=0.002, spread_gain=SPREAD_GAIN,
                           max_indent_m=self.depth, reform_rate=REFORM)
        self._tile_xy: Optional[np.ndarray] = None
        self._tile_h: Optional[np.ndarray] = None
        self._tile_ops: list = []
        self._finger_prims: list = []
        self.pressing = False            # part-press coupling gate (lip strategy)
        self.press_foot = None           # phantom foot (px, py, hw, pen)
        self._part_paths_fn = None       # callable → list of live part prim paths

    # ── build ────────────────────────────────────────────────────────────────
    def build(self, gripper_body_prims: List, finger_prims: List) -> "SoftRig":
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
        stage = self.stage
        self._finger_prims = list(finger_prims)

        # (1) filter every gripper link from the sink colliders (platform slab +,
        # on the platform scene, the table-top collider beneath it)
        glinks = [p.GetPath() for p in gripper_body_prims] or \
                 [p.GetPath() for p in finger_prims]
        for pp in self.platform_paths:
            plat = stage.GetPrimAtPath(pp)
            if plat and plat.IsValid() and glinks:
                rel = UsdPhysics.FilteredPairsAPI.Apply(plat).CreateFilteredPairsRel()
                for t in glinks:
                    rel.AddTarget(t)

        # (2) rigid hard stop `depth` below the surface (NOT filtered)
        hs_path = "/World/SortHardStop"
        if stage.GetPrimAtPath(hs_path).IsValid():
            stage.RemovePrim(hs_path)
        top = self.surface_z - self.depth
        cube = UsdGeom.Cube.Define(stage, hs_path)
        cube.GetSizeAttr().Set(1.0)
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetTranslate(
            Gf.Vec3d(self.centre[0], self.centre[1], top - HARDSTOP_THICK / 2.0))
        UsdGeom.XformCommonAPI(cube.GetPrim()).SetScale(
            Gf.Vec3f(0.8, 1.2, HARDSTOP_THICK))
        cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if PhysxSchema is not None:
            pc = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            pc.CreateContactOffsetAttr().Set(0.001)
            pc.CreateRestOffsetAttr().Set(0.0)

        # (3) kinematic support tiles over the pick zone (parts ride the dent)
        parent = "/World/SortTiles"
        if stage.GetPrimAtPath(parent).IsValid():
            stage.RemovePrim(parent)
        UsdGeom.Xform.Define(stage, parent)
        tmat_path = parent + "/TileMat"
        UsdShade.Material.Define(stage, tmat_path)
        tm = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(tmat_path))
        tm.CreateStaticFrictionAttr().Set(TILE_FRICTION)
        tm.CreateDynamicFrictionAttr().Set(TILE_FRICTION)
        tm.CreateRestitutionAttr().Set(0.0)
        if PhysxSchema is not None:
            PhysxSchema.PhysxMaterialAPI.Apply(
                stage.GetPrimAtPath(tmat_path)).CreateFrictionCombineModeAttr().Set("min")
        tmat = UsdShade.Material(stage.GetPrimAtPath(tmat_path))

        cx, cy = self.centre
        span, cell = TILE_SPAN, TILE_CELL
        n = max(2, int(round(span / cell)))
        half = span / 2.0
        w = cell * 0.98
        tile_xy, ops = [], []
        for i in range(n):
            for j in range(n):
                tx = cx - half + span * (i + 0.5) / n
                ty = cy - half + span * (j + 0.5) / n
                tile_xy.append((tx, ty))
                p = f"{parent}/tile_{i}_{j}"
                body = UsdGeom.Xform.Define(stage, p)
                op = UsdGeom.Xformable(body.GetPrim()).AddTranslateOp()
                op.Set(Gf.Vec3d(tx, ty, self.surface_z - TILE_HALF))
                ops.append(op)
                geo = UsdGeom.Cube.Define(stage, p + "/geo")
                geo.GetSizeAttr().Set(1.0)
                UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(
                    Gf.Vec3f(w, w, 2 * TILE_HALF))
                UsdGeom.Imageable(geo.GetPrim()).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
                if PhysxSchema is not None:
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
                    pc.CreateContactOffsetAttr().Set(0.002)
                    pc.CreateRestOffsetAttr().Set(0.0)
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(True)
                UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
                    tmat, UsdShade.Tokens.weakerThanDescendants, "physics")
                rel = UsdPhysics.FilteredPairsAPI.Apply(
                    geo.GetPrim()).CreateFilteredPairsRel()
                for gl in glinks:
                    rel.AddTarget(gl)
        self._tile_xy = np.array(tile_xy, dtype=np.float32)
        self._tile_h = np.full(self._tile_xy.shape[0], self.surface_z, dtype=np.float32)
        self._tile_ops = ops
        print(f"[soft] rig built: {n}x{n} tiles, hard stop {self.depth * 1000:.0f}mm, "
              f"ik floor {self.ik_depth * 1000:.0f}mm, {len(glinks)} gripper links filtered",
              flush=True)
        return self

    def filter_part(self, part_path: str) -> None:
        """A spawned part rides the tiles: filter it from the rigid sink colliders."""
        from pxr import UsdPhysics, Sdf
        for pp in self.platform_paths:
            plat = self.stage.GetPrimAtPath(pp)
            if plat and plat.IsValid():
                UsdPhysics.FilteredPairsAPI.Apply(plat).CreateFilteredPairsRel(
                ).AddTarget(Sdf.Path(part_path))

    def set_parts_source(self, fn) -> None:
        """Callable returning the live part prim paths (for part-press coupling)."""
        self._part_paths_fn = fn

    # ── per-step ─────────────────────────────────────────────────────────────
    def _feet(self):
        """Fingertip dent feet + optional part-press coupling + phantom press foot.
        (adapter.py::_soft_feet)"""
        from pxr import UsdGeom
        feet = []
        try:
            bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            part_bbs = []
            if self.pressing and self._part_paths_fn is not None:
                for pp in self._part_paths_fn():
                    prim = self.stage.GetPrimAtPath(pp)
                    if prim and prim.IsValid():
                        b = bc.ComputeWorldBound(prim).ComputeAlignedBox()
                        part_bbs.append((b.GetMin(), b.GetMax()))
            for p in self._finger_prims:
                if not p or not p.IsValid():
                    continue
                wb = bc.ComputeWorldBound(p).ComputeAlignedBox()
                mn, mx = wb.GetMin(), wb.GetMax()
                cx = 0.5 * (float(mn[0]) + float(mx[0]))
                cy = 0.5 * (float(mn[1]) + float(mx[1]))
                hw = max(0.5 * float(mx[0] - mn[0]),
                         0.5 * float(mx[1] - mn[1]), FOOT_HW)
                pen = max(0.0, self.surface_z - float(mn[2]))
                for (bmn, bmx) in part_bbs:
                    if (float(mn[0]) < float(bmx[0]) and float(mx[0]) > float(bmn[0])
                            and float(mn[1]) < float(bmx[1])
                            and float(mx[1]) > float(bmn[1])):
                        overlap = float(bmx[2]) - float(mn[2])
                        # STRICT contact only (overlap > 0): relaxed variants
                        # (probe_press_grasp runs 8–17, 2026-07-03) either fired
                        # the dent early — the washer slid into the forming
                        # valley before contact — or killed the flip-and-LEAN
                        # that sustains the stage-1 lip (A/B 0038-press_lift:
                        # peak preserved, sustain lost). Do not soften this gate.
                        if overlap > 0.0:
                            sink = max(0.0, self.surface_z - float(bmn[2]))
                            pen = max(pen, min(overlap + sink, self.depth))
                feet.append((cx, cy, hw, pen))
        except Exception:
            feet = []
        if self.press_foot is not None:
            feet.append(tuple(self.press_foot))
        return feet

    def step(self) -> None:
        """Dent/reform: drive the kinematic tile tops to the SoftPad field."""
        if self._tile_xy is None:
            return
        from pxr import Gf
        feet = self._feet()
        Zt = self.pad.target_field(self._tile_xy[:, 0], self._tile_xy[:, 1],
                                   self.surface_z, feet)
        self._tile_h = self.pad.relax(self._tile_h, Zt.astype(np.float32))
        for k, op in enumerate(self._tile_ops):
            op.Set(Gf.Vec3d(float(self._tile_xy[k, 0]), float(self._tile_xy[k, 1]),
                            float(self._tile_h[k]) - TILE_HALF))

    def max_dent_mm(self) -> float:
        if self._tile_h is None:
            return 0.0
        return float((self.surface_z - np.min(self._tile_h)) * 1000.0)
