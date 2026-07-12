"""
Compliant elastic work surface — a Winkler foundation, the diagram-02 model.

Unlike `soft_rig`'s KINEMATIC tiles (driven each step to an analytic dent field,
so they don't respond to contact — bistable, snaps), this is a grid of small
DYNAMIC rigid tiles, each on a vertical prismatic joint sprung to its rest
height (linear DriveAPI: stiffness k, damping c, generous downward travel).

Press one side and the surface yields IN PROPORTION to the load, so a rigid part
tilts smoothly instead of snapping. PhysX integrates the springs — there is NO
per-step scripting (a plus for headless stability and for parallel scale).

Reuses the exact prismatic + linear-drive construct already proven on the
gripper fingers (gripper.py:182–202).

    found = SpringFoundation(stage, centre_xy, surf_z, stiffness=300.0)
    found.build()
    # ... spawn a part on top, step physics ...
    found.max_deflection_mm()
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def neoprene(E_eff: float, t: float, nu: float = 0.25, cell: float = 0.005):
    """Map pad material physics → discrete foundation parameters.

    A tile column compresses like a patch of the pad:  k_cell = E'·cell²/t.
    The Pasternak shear layer between columns:         k_link = G·t
    (independent of cell size), G = E/(2(1+ν)). The emergent dent decay
    length is ℓ = cell·√(k_link/k_cell) ≈ pad thickness for real foam/rubber
    — ℓ is what the eye reads as foam (ℓ≈t) vs gel (ℓ→0). See
    data/reports/2026-07-04_materials.md §4.

    Returns dict(stiffness, couple, ell) for SpringFoundation(**...).
    """
    G = E_eff / (2.0 * (1.0 + nu))
    k_cell = E_eff * cell * cell / t
    # SOLVER CORRECTION α: PhysX TGS under-enforces stiff drive CHAINS — the
    # measured dish is ~2x narrower than the Pasternak prediction (constant
    # effective-ratio deficit ≈ 4x across R3..R10, bar-load probe 2026-07-04,
    # unchanged by solver-iteration bumps). Configure couple = α·G·t so the
    # MEASURED profile matches the physical target shape.
    ALPHA = 4.0
    k_link = ALPHA * G * t
    ell = cell * float(np.sqrt((k_link / ALPHA) / k_cell))   # physical ℓ
    return dict(stiffness=k_cell, couple=k_link, ell=ell)


class SpringFoundation:
    def __init__(self, stage, centre_xy, surf_z: float,
                 span: float = 0.060, cell: float = 0.004,
                 stiffness: float = 300.0, damping: float = 6.0,
                 tile_mass: float = 0.006, travel: float = 0.014,
                 friction: float = 0.9, tile_h: float = 0.010,
                 couple: float = 0.0, couple_damp: float = 0.0,
                 couple_falloff: float = 0.0, couple_cutoff: float = 0.10,
                 rotz_deg: float = 0.0,
                 anchor: str = "",
                 visible: bool = True, tile_color=(0.80, 0.66, 0.45),
                 tile_color2=(0.55, 0.43, 0.26),
                 parent: str = "/World/Foundation"):
        self.stage = stage
        self.surf_z = float(surf_z)
        self.centre = (float(centre_xy[0]), float(centre_xy[1]))
        self.span = float(span)
        self.cell = float(cell)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.tile_mass = float(tile_mass)
        self.travel = float(travel)
        self.friction = float(friction)
        self.tile_h = float(tile_h)
        # shear coupling between neighbouring tiles (Pasternak layer): 0 = pure
        # independent Winkler springs (dimple under the load); >0 = the surface
        # behaves like an elastic sheet/foam and a load spreads into a smooth DISH.
        self.couple = float(couple)
        self.couple_damp = float(couple_damp if couple_damp else damping)
        # drop-off over the plane: links also reach neighbours-of-neighbours with
        # strength couple * falloff^(r-1) (r = centre distance in cells). Links
        # weaker than couple_cutoff * couple are dropped, so the influence radius
        # is finite. falloff=0 keeps the original nearest-neighbour-only layer.
        self.couple_falloff = float(couple_falloff)
        self.couple_cutoff = float(couple_cutoff)
        # In-plane rotation of the tile lattice about the centre (radians). Used
        # to present the fixed-frame gripper a differently-oriented slot lattice
        # per rig — an EE-roll / approach-angle robustness axis (the washer is a
        # symmetric disc, so rotating the surface ≡ rolling the EE). Springs are
        # vertical and coupling is grid-topological, so rotation only moves the
        # tile (x,y); nothing else changes.
        self.rotz = float(np.radians(rotz_deg))
        # Isaac Lab cloning: joints anchored to WORLD carry world-frame
        # localPos0 and break when the env template is cloned to new origins.
        # Pass `anchor` = path of a static rigid body at the env origin; the
        # vertical springs then attach to IT with env-local coordinates and
        # the whole foundation clones cleanly.
        self.anchor = anchor
        self.visible = bool(visible)
        self.tile_color = tuple(tile_color)
        self.tile_color2 = tuple(tile_color2)
        self.parent = parent
        self.tile_paths: List[str] = []
        self._tile_xy: Optional[np.ndarray] = None
        self._rest_top = self.surf_z
        self._grid_n = 0
        self._drives: list = []
        self._couple_drives: list = []   # (DriveAPI, base_stiffness, base_damping)
        self._targets: Optional[np.ndarray] = None   # last authored drive targets

    def build(self) -> "SpringFoundation":
        from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
        try:
            from pxr import PhysxSchema
        except ImportError:
            PhysxSchema = None
        stage = self.stage
        if stage.GetPrimAtPath(self.parent).IsValid():
            stage.RemovePrim(self.parent)
        self.tile_paths = []
        self._drives = []
        self._couple_drives = []
        self._targets = None
        UsdGeom.Xform.Define(stage, self.parent)

        # tile-top friction material (holds the part laterally so it PIVOTS
        # rather than squirting; the pressing tip is low-friction separately)
        mat_path = self.parent + "/TileMat"
        UsdShade.Material.Define(stage, mat_path)
        m = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
        m.CreateStaticFrictionAttr(self.friction)
        m.CreateDynamicFrictionAttr(self.friction)
        m.CreateRestitutionAttr(0.0)
        mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))

        n = max(2, int(round(self.span / self.cell)))
        half = self.span / 2.0
        w = self.cell * 0.92
        cx, cy = self.centre
        tile_xy = []
        _cr, _sr = np.cos(self.rotz), np.sin(self.rotz)
        for i in range(n):
            for j in range(n):
                tx = cx - half + self.span * (i + 0.5) / n
                ty = cy - half + self.span * (j + 0.5) / n
                if self.rotz:                       # rotate the lattice about centre
                    lx, ly = tx - cx, ty - cy
                    tx = cx + lx * _cr - ly * _sr
                    ty = cy + lx * _sr + ly * _cr
                tile_xy.append((tx, ty))
                p = f"{self.parent}/t_{i}_{j}"
                # RigidBody on an UNSCALED Xform + scaled child collider (PhysX
                # bakes scale into the body frame otherwise — parts.py lesson)
                body = UsdGeom.Xform.Define(stage, p)
                cz = self.surf_z - self.tile_h / 2.0
                UsdGeom.XformCommonAPI(body.GetPrim()).SetTranslate(
                    Gf.Vec3d(tx, ty, cz))
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(self.tile_mass)
                if PhysxSchema is not None:
                    # strong coupling (k_link >> k_cell) forms stiff drive CHAINS;
                    # default iterations under-converge them and the dish comes out
                    # ~2x narrower than the Pasternak prediction — give the solver
                    # more position iterations on the tiles.
                    rbx = PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim())
                    rbx.CreateSolverPositionIterationCountAttr(16)
                    rbx.CreateSolverVelocityIterationCountAttr(2)

                geo = UsdGeom.Cube.Define(stage, p + "/geo")
                geo.GetSizeAttr().Set(1.0)
                UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(
                    Gf.Vec3f(w, w, self.tile_h))
                if self.visible:
                    col = self.tile_color if (i + j) % 2 == 0 else self.tile_color2
                    geo.CreateDisplayColorAttr([Gf.Vec3f(*col)])
                else:
                    UsdGeom.Imageable(geo.GetPrim()).MakeInvisible()
                UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
                if PhysxSchema is not None:
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
                    pc.CreateContactOffsetAttr().Set(0.0015)
                    pc.CreateRestOffsetAttr().Set(0.0)
                UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics")

                # vertical prismatic spring to WORLD (default) or to the
                # per-env ANCHOR body (Isaac Lab clone-safe; env-local coords)
                jt = UsdPhysics.PrismaticJoint.Define(stage, p + "/spring")
                jt.CreateAxisAttr(UsdPhysics.Tokens.z)
                jt.CreateBody1Rel().SetTargets([Sdf.Path(p)])
                if self.anchor:
                    jt.CreateBody0Rel().SetTargets([Sdf.Path(self.anchor)])
                    ap = stage.GetPrimAtPath(self.anchor)
                    at = UsdGeom.Xformable(ap).ComputeLocalToWorldTransform(0)
                    a0 = at.ExtractTranslation()
                    jt.CreateLocalPos0Attr(Gf.Vec3f(
                        float(tx - a0[0]), float(ty - a0[1]), float(cz - a0[2])))
                else:
                    jt.CreateLocalPos0Attr(
                        Gf.Vec3f(float(tx), float(ty), float(cz)))
                jt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                jt.CreateLowerLimitAttr(-self.travel)   # yields DOWN by travel
                jt.CreateUpperLimitAttr(0.0005)         # barely rises past rest
                drv = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "linear")
                drv.CreateTypeAttr("force")
                drv.CreateMaxForceAttr(1.0e6)
                drv.CreateTargetPositionAttr(0.0)       # rest height
                drv.CreateStiffnessAttr(self.stiffness)
                drv.CreateDampingAttr(self.damping)
                self._drives.append(drv)
                self.tile_paths.append(p)
        self._tile_xy = np.array(tile_xy, dtype=np.float32)
        self._grid_n = n

        # ── shear coupling (Pasternak layer) ──────────────────────────────────
        # Link each tile to its right and up neighbour with a generic (D6) joint
        # that leaves every DOF free EXCEPT a Z drive pulling their heights equal
        # (target relative-Z = 0). This is a soft spring between adjacent tile
        # tops, so a load no longer sinks one tile alone — it drags the neighbours
        # down too and the surface forms a smooth dish, like real foam/rubber.
        n_couple = 0
        if self.couple > 0.0:
            # offset table (half-plane, so each unordered pair links once) with
            # per-offset strength from the drop-off law. falloff=0 → 0**0=1 keeps
            # the two r=1 orthogonal links and kills everything further out.
            offs = []
            RMAX = 4
            for di in range(0, RMAX + 1):
                for dj in range(-RMAX, RMAX + 1):
                    if di == 0 and dj <= 0:
                        continue
                    r = float(np.hypot(di, dj))
                    if r > RMAX + 1e-6:
                        continue
                    s = self.couple_falloff ** (r - 1.0) if r > 1.0 else 1.0
                    if s < self.couple_cutoff:
                        continue
                    offs.append((di, dj, s))

            def _path(i, j):
                return f"{self.parent}/t_{i}_{j}"
            for i in range(n):
                for j in range(n):
                    for di, dj, s in offs:
                        ni, nj = i + di, j + dj
                        if not (0 <= ni < n and 0 <= nj < n):
                            continue
                        a, b = _path(i, j), _path(ni, nj)
                        jp = f"{a}/couple_{di}_{dj}".replace("-", "m")
                        jt = UsdPhysics.Joint.Define(stage, jp)
                        jt.CreateBody0Rel().SetTargets([Sdf.Path(a)])
                        jt.CreateBody1Rel().SetTargets([Sdf.Path(b)])
                        # frames at each body origin (tiles never rotate → the
                        # transZ drive measures zB − zA, which is 0 at rest)
                        jt.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                        jt.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
                        jt.CreateExcludeFromArticulationAttr(True)
                        drv = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "transZ")
                        drv.CreateTypeAttr("force")
                        drv.CreateMaxForceAttr(1.0e6)
                        drv.CreateTargetPositionAttr(0.0)
                        drv.CreateStiffnessAttr(self.couple * s)
                        drv.CreateDampingAttr(self.couple_damp * s)
                        self._couple_drives.append(
                            (drv, self.couple * s, self.couple_damp * s))
                        n_couple += 1

        print(f"[foundation] {n}x{n} sprung tiles, k={self.stiffness} N/m "
              f"c={self.damping} cell={self.cell*1000:.0f}mm travel={self.travel*1000:.0f}mm "
              f"friction={self.friction} couple={self.couple} "
              f"falloff={self.couple_falloff}/{self.couple_cutoff} ({n_couple} links)",
              flush=True)
        return self

    def retune(self, k_scale: float = 1.0, c_scale: float = 1.0,
               couple_scale: float = 1.0):
        """Re-author the spring/coupling drive gains — a DIFFERENT material,
        same lattice. Cheap USD attr writes (no rebuild), so it can run per
        episode for material domain-randomization. Scales are relative to the
        material the foundation was built with (absolute, not compounding)."""
        for drv in self._drives:
            drv.GetStiffnessAttr().Set(self.stiffness * float(k_scale))
            drv.GetDampingAttr().Set(self.damping * float(c_scale))
        for drv, k0, c0 in self._couple_drives:
            drv.GetStiffnessAttr().Set(k0 * float(couple_scale))
            drv.GetDampingAttr().Set(c0 * float(couple_scale))
        return self

    # ── readout ──────────────────────────────────────────────────────────────
    def tile_tops(self) -> np.ndarray:
        """Current world-Z of every tile top (surf_z − deflection)."""
        from pxr import UsdGeom
        xc = UsdGeom.XformCache()
        z = np.empty(len(self.tile_paths), dtype=np.float32)
        for k, p in enumerate(self.tile_paths):
            M = xc.GetLocalToWorldTransform(self.stage.GetPrimAtPath(p))
            z[k] = float(M.ExtractTranslation()[2]) + self.tile_h / 2.0
        return z

    def max_deflection_mm(self) -> float:
        return float((self.surf_z - np.min(self.tile_tops())) * 1000.0)

    def deflection_field_mm(self) -> np.ndarray:
        """n×n grid (indexed [i,j]) of downward deflection in mm (>=0 down)."""
        n = self._grid_n
        d = (self.surf_z - self.tile_tops()) * 1000.0
        return d.reshape(n, n)

    def dish_profile_mm(self) -> np.ndarray:
        """Deflection along the centre row (i.e. the surface cross-section)."""
        f = self.deflection_field_mm()
        return f[:, f.shape[1] // 2]

    def spread_ratio(self) -> float:
        """How wide the dent is: mean deflection / peak deflection (0..1).
        Independent Winkler ≈ small (only the loaded tile moves); a well-coupled
        elastic sheet ≈ large (the whole surface dishes)."""
        f = self.deflection_field_mm()
        pk = float(np.max(f))
        return float(np.mean(f) / pk) if pk > 1e-6 else 0.0

    # ── proximity leveling ───────────────────────────────────────────────────
    def level_targets(self, bodies, ell: float, margin: float = 0.001) -> None:
        """Conform the surface AHEAD of contact (report §4, fig/02): bias tile
        spring TARGETS toward the underside of nearby bodies so approaching /
        dragging objects meet a ramp instead of a one-cell wall. Springs stay
        springs — contact forces remain real physics.

        bodies: iterable of (cx, cy, half_x, half_y, bottom_z) AABB footprints.
        Only tiles OUTSIDE a footprint are leveled (d > 0): interior tiles keep
        full spring support, so a resting part cannot chase its own targets
        down (runaway sink). Weight exp(−d²/2ℓ²), level to bottom_z + margin,
        never above rest, capped at 95% travel. Pass bodies=[] to relax.
        """
        xy = self._tile_xy
        if xy is None or not self._drives:
            return
        drop = np.zeros(len(self._drives), dtype=np.float64)
        for (bx, by, hx, hy, bz) in bodies:
            if bz > self.surf_z + 0.002:       # body nowhere near the surface
                continue
            depth = self.surf_z - (bz + margin)
            if depth <= 0.0:
                continue
            dx = np.maximum(np.abs(xy[:, 0] - bx) - hx, 0.0)
            dy = np.maximum(np.abs(xy[:, 1] - by) - hy, 0.0)
            d2 = dx * dx + dy * dy
            w = np.exp(-d2 / (2.0 * ell * ell))
            w[d2 <= 0.0] = 0.0                 # interior: contact does the work
            drop = np.maximum(drop, depth * w)
        np.clip(drop, 0.0, self.travel * 0.95, out=drop)
        if self._targets is None:
            self._targets = np.zeros_like(drop)
        changed = np.abs(drop - self._targets) > 5e-5
        for i in np.nonzero(changed)[0]:
            self._drives[i].GetTargetPositionAttr().Set(float(-drop[i]))
        self._targets[changed] = drop[changed]


class SurfaceSkin:
    """A non-colliding VISUAL mesh draped over the foundation's tile tops that
    follows the deformation each frame — a smooth continuous 'foam skin' instead
    of the discrete physics tiles (hide the tiles with SpringFoundation(visible=
    False) and show only this). Purely cosmetic: no CollisionAPI, no rigid body,
    so it never touches the physics — the part/finger pass through it visually.

        skin = SurfaceSkin(found).build()
        # each render frame, after stepping physics:
        skin.update()
    """

    def __init__(self, foundation: "SpringFoundation",
                 path: str = "/World/FoundationSkin",
                 color=(0.86, 0.62, 0.45), lift: float = 0.0005,
                 subdivide: bool = True):
        self.f = foundation
        self.stage = foundation.stage
        self.path = path
        self.color = tuple(color)
        self.lift = float(lift)       # tiny raise so the skin sits above tile tops
        self.subdivide = bool(subdivide)
        self._mesh = None
        self._n = 0

    def build(self) -> "SurfaceSkin":
        from pxr import UsdGeom, Gf
        n = self.f._grid_n
        self._n = n
        xy = self.f._tile_xy.reshape(n, n, 2)
        pts = [Gf.Vec3f(float(xy[i, j, 0]), float(xy[i, j, 1]),
                        float(self.f.surf_z + self.lift))
               for i in range(n) for j in range(n)]
        counts, idx = [], []
        for i in range(n - 1):
            for j in range(n - 1):
                a, b = i * n + j, i * n + (j + 1)
                c, d = (i + 1) * n + (j + 1), (i + 1) * n + j
                counts.append(4)
                idx += [a, b, c, d]     # quads; Catmull-Clark smooths them
        mesh = UsdGeom.Mesh.Define(self.stage, self.path)
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr("catmullClark" if self.subdivide else "none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*self.color)])
        # planar UVs, 10mm period — lets a grid texture (workshop-site look)
        # tile in real-world millimetres regardless of bed size
        from pxr import UsdShade as _US, Sdf as _Sdf
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", _Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec2f(float(xy[i, j, 0]) / 0.010,
                         float(xy[i, j, 1]) / 0.010)
                for i in range(n) for j in range(n)])
        mesh.CreateDoubleSidedAttr(True)
        self._mesh = mesh
        return self

    def update(self) -> None:
        """Refresh the skin's vertex heights from the current tile tops."""
        from pxr import Gf
        n = self._n
        xy = self.f._tile_xy.reshape(n, n, 2)
        tops = self.f.tile_tops().reshape(n, n)
        pts = [Gf.Vec3f(float(xy[i, j, 0]), float(xy[i, j, 1]),
                        float(tops[i, j] + self.lift))
               for i in range(n) for j in range(n)]
        self._mesh.GetPointsAttr().Set(pts)
