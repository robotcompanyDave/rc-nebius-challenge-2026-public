"""
Procedural parametric fastener parts — nuts, bolts AND washers, no asset pack.

The private platform used to spawn NVIDIA Factory USD props from a hardcoded
Windows path; both repos now generate the geometry in code (this file is kept in
sync with rc-remote-platform/targets/ur10e/parts.py):

  • hex nut  — DIN/ISO across-flats + thickness per M-size, cosmetic bore
  • bolt     — hex head + Ø shank, length 30–50 mm (randomised per spawn)
  • washer   — ISO 7089 flat washer (true annulus mesh: OD/ID/thickness per size)

and wrap each part in a SHAPE-ACCURATE physics treatment: convex-hull collider on
the hex body / washer ring + a capsule on the bolt shank, geometry-derived steel
mass (an M6 washer is ~1 g, an M12 bolt ~50 g — that spread is what the grasp
physics must learn), contact-offset/rest-offset, and a realistic steel friction
material (the grasp hold comes from the high-friction gripper pad, not the part).
Seat bbox-min 1 mm above the platform, no drop — from `adapter.py::_spawn_nutsbolts`.

The working span for the soft-surface training effort is M12 down to M6
(`WORK_SIZES`); the flat washer (1.6–2.5 mm thick) is the archetypal compliant-
surface pick that motivates the press-for-lip strategy.

Import note: safe to import where `pxr` is available (Isaac's python OR a plain
`usd-core` install for offline authoring). `PhysxSchema` is Isaac-only, so it is
imported lazily and its calls are skipped when absent (offline validation).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
# NOTE: `pxr` is imported lazily inside each function below. In Isaac's python the
# USD libraries are not on sys.path until a SimulationApp has been constructed, and
# this module is imported before that happens (see the boot-order note in
# sim_env.py). Every function here runs only after the app exists.

# ── physical constants (ported from adapter.py:252) ────────────────────────
MASS_KG = 0.02                 # fallback only; real mass is geometry × density (below)
STEEL_DENSITY = 7850.0         # kg/m^3 — the parts are steel; per-part mass ∝ volume
STEEL_RGB = (0.62, 0.64, 0.68)
GRASP_FRICTION = 0.6           # the part is STEEL: realistic steel-on-steel/table
                               # friction so parts aren't sticky on the bench or each
                               # other. The grasp HOLD comes from the high-friction
                               # gripper PAD (gripper.GripperParams.pad_friction), not
                               # from the part. (Was 1.4 — made every surface grippy.)
CONTACT_OFFSET = 0.004         # PhysX low contact offset (thin part on a thin slab)
REST_OFFSET = 0.0


def _hex_area(af: float) -> float:
    """Cross-section area of a hex prism with across-flats `af`."""
    R = af / math.sqrt(3.0)                        # circumradius
    return (3.0 * math.sqrt(3.0) / 2.0) * R * R


def _part_mass(kind: str, dims: dict, shank_len: float) -> float:
    """Realistic steel mass from the ACTUAL geometry (vs a flat 20 g for all sizes).
    A real workshop M8 nut is ~5 g, an M12 washer ~6 g, an M16 bolt ~40 g — that
    spread matters for the physics the sim is meant to teach."""
    if kind == "bolt":
        head_h = max(0.006, dims["nut_h"] * 0.8)
        vol = (_hex_area(dims["af"]) * head_h
               + math.pi * (dims["shank_d"] / 2.0) ** 2 * shank_len)
    elif kind == "washer":
        vol = (math.pi / 4.0) * (dims["w_od"] ** 2 - dims["w_id"] ** 2) * dims["w_t"]
    else:
        bore_r = dims["shank_d"] / 2.0             # the bore removes material
        vol = (_hex_area(dims["af"]) - math.pi * bore_r ** 2) * dims["nut_h"]
    return max(vol * STEEL_DENSITY, 0.001)

# ── M-size table (metres) ───────────────────────────────────────────────────
# af      = across-flats (DIN 934 family — matches the tuned M12=19mm)
# nut_h   = nut thickness            shank_d = bolt shank / nominal Ø
# w_od / w_id / w_t = ISO 7089 flat-washer outer Ø / inner Ø / thickness
_MSIZE = {
    "m4":  dict(af=0.007,  nut_h=0.0032, shank_d=0.004,
                w_od=0.009, w_id=0.0043, w_t=0.0008),
    "m6":  dict(af=0.010,  nut_h=0.0050, shank_d=0.006,
                w_od=0.012, w_id=0.0064, w_t=0.0016),
    "m8":  dict(af=0.013,  nut_h=0.0065, shank_d=0.008,
                w_od=0.016, w_id=0.0084, w_t=0.0016),
    "m10": dict(af=0.017,  nut_h=0.0080, shank_d=0.010,
                w_od=0.020, w_id=0.0105, w_t=0.0020),
    "m12": dict(af=0.019,  nut_h=0.0100, shank_d=0.012,
                w_od=0.024, w_id=0.0130, w_t=0.0025),
    "m16": dict(af=0.024,  nut_h=0.0130, shank_d=0.016,
                w_od=0.030, w_id=0.0170, w_t=0.0030),
    "m20": dict(af=0.030,  nut_h=0.0160, shank_d=0.020,
                w_od=0.037, w_id=0.0220, w_t=0.0030),
}
DEFAULT_SIZE = "m12"
KINDS = ("nut", "bolt", "washer")
# The working span for the soft-surface sort/training effort: M12 down to M6.
WORK_SIZES = ("m6", "m8", "m10", "m12")


def size_dims(size: str) -> dict:
    return _MSIZE.get(size, _MSIZE[DEFAULT_SIZE])


def part_grip_width(kind: str, dims: dict) -> float:
    """Jaw-relevant width (m): what the closed jaw spans on the natural side grip."""
    if kind == "washer":
        return dims["w_od"]
    return dims["af"]


def part_height_flat(kind: str, dims: dict) -> float:
    """Resting height (m) of the part lying FLAT — the ledge the fingertip must get
    under/around. A flat M6 washer is 1.6 mm: the soft-surface case."""
    if kind == "washer":
        return dims["w_t"]
    if kind == "bolt":
        return dims["af"] * 0.866
    return dims["nut_h"]


# Pose classes → rotation about local X (degrees). Mirrors adapter.py pose
# scenarios: a nut/washer lies flat at 0°, a bolt lies flat at 90° (shaft
# horizontal), "standing" stands a part on end / a bolt head-down. A washer
# on its rim is unstable and usually settles flat — itself a good test case.
_POSE_ROTX = {
    ("nut", "flat"): 0.0,
    ("nut", "on-side"): 90.0,
    ("nut", "standing"): 90.0,
    ("bolt", "flat"): 90.0,     # shaft horizontal
    ("bolt", "on-side"): 90.0,
    ("bolt", "standing"): 180.0,  # head-down, shaft vertical
    ("washer", "flat"): 0.0,
    ("washer", "on-side"): 90.0,
    ("washer", "standing"): 90.0,
}
POSE_CLASSES = ("flat", "on-side", "standing", "random")


@dataclass
class PartSpec:
    kind: str          # "nut" | "bolt" | "washer"
    size: str          # "m6" … "m12" (any _MSIZE key)
    pose: str          # "flat" | "on-side" | "standing" | "random"
    xy: tuple          # (x, y) world position on the platform
    rotz_deg: float    # yaw about world Z


# ── mesh builders ──────────────────────────────────────────────────────────
def _hex_prism_points(af: float, height: float) -> list:
    """12 points of a capped hexagonal prism, axis along +Z, centred at origin.

    Flats face the X and Y axes (vertices at 30°+k·60°) so the nut's flat sides
    line up with the world axes at yaw 0 — matching the jaw flat-alignment in the
    controller (`_grasp_R`, FLAT_OFFSET=0)."""
    circumradius = af / math.sqrt(3.0)           # AF = 2·R·cos30° → R = AF/√3
    hz = height / 2.0
    pts = []
    for z in (hz, -hz):
        for k in range(6):
            a = math.radians(30.0 + 60.0 * k)
            pts.append((circumradius * math.cos(a), circumradius * math.sin(a), z))
    return pts


def _hex_prism_faces() -> tuple:
    """(faceVertexCounts, faceVertexIndices) for the 12-point capped hex prism."""
    counts = [6, 6] + [4] * 6                      # top hex, bottom hex, 6 side quads
    top = [0, 1, 2, 3, 4, 5]
    bottom = [11, 10, 9, 8, 7, 6]                  # reversed for outward normal
    idx = list(top) + list(bottom)
    for i in range(6):
        j = (i + 1) % 6
        idx += [i, j, 6 + j, 6 + i]
    return counts, idx


def _define_hex_mesh(stage, path: str, af: float, height: float) -> "UsdGeom.Mesh":
    from pxr import UsdGeom, Gf
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = _hex_prism_points(af, height)
    counts, idx = _hex_prism_faces()
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


def _define_annulus_mesh(stage, path: str, od: float, idm: float, t: float,
                         segs: int = 24) -> "UsdGeom.Mesh":
    """True flat-washer ring: outer wall, inner bore wall, flat top/bottom rings."""
    from pxr import UsdGeom, Gf
    mesh = UsdGeom.Mesh.Define(stage, path)
    ro, ri, hz = od / 2.0, idm / 2.0, t / 2.0
    pts = []
    # ring order: [top-outer 0..n-1][top-inner n..2n-1][bot-outer][bot-inner]
    for z in (hz, -hz):
        for r in (ro, ri):
            for k in range(segs):
                a = 2.0 * math.pi * k / segs
                pts.append((r * math.cos(a), r * math.sin(a), z))
    TO, TI, BO, BI = 0, segs, 2 * segs, 3 * segs
    counts, idx = [], []
    for k in range(segs):
        j = (k + 1) % segs
        counts += [4, 4, 4, 4]
        idx += [TO + k, TO + j, TI + j, TI + k]          # top ring (normal +Z)
        idx += [BI + k, BI + j, BO + j, BO + k]          # bottom ring (normal -Z)
        idx += [TO + j, TO + k, BO + k, BO + j]          # outer wall (outward)
        idx += [TI + k, TI + j, BI + j, BI + k]          # inner bore wall (inward)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return mesh


# ── materials ──────────────────────────────────────────────────────────────
def ensure_steel_material(stage) -> "UsdShade.Material":
    """Cached steel UsdPreviewSurface (adapter.py:2473)."""
    from pxr import UsdGeom, UsdShade, Sdf, Gf
    mat_path = "/World/Looks/NutBoltSteel"
    existing = stage.GetPrimAtPath(mat_path)
    if existing.IsValid():
        return UsdShade.Material(existing)
    UsdGeom.Scope.Define(stage, "/World/Looks")
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*STEEL_RGB))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def ensure_grasp_physics_material(stage, friction: float = GRASP_FRICTION):
    """Cached high-friction PhysX material so a closed jaw holds the part by
    friction (adapter.py:_ensure_grasp_material). No-op offline (no PhysxSchema)."""
    from pxr import UsdShade, UsdPhysics
    try:
        from pxr import PhysxSchema
    except ImportError:
        return None
    mat_path = "/World/Looks/NutBoltGrip"
    prim = stage.GetPrimAtPath(mat_path)
    if not prim.IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        pmat = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pmat.CreateStaticFrictionAttr().Set(friction)
        pmat.CreateDynamicFrictionAttr().Set(friction)
        pmat.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        prim = mat.GetPrim()
    return UsdShade.Material(prim)


# ── part construction ────────────────────────────────────────────────────────
def _build_nut(stage, path: str, dims: dict) -> None:
    from pxr import UsdGeom
    _define_hex_mesh(stage, path + "/body", dims["af"], dims["nut_h"])
    # cosmetic bore (visual only; the hull collider ignores it)
    bore = UsdGeom.Cylinder.Define(stage, path + "/bore")
    bore.CreateAxisAttr(UsdGeom.Tokens.z)
    bore.CreateRadiusAttr(dims["shank_d"] / 2.0)
    bore.CreateHeightAttr(dims["nut_h"] * 1.05)


def _build_bolt(stage, path: str, dims: dict, shank_len: float) -> None:
    from pxr import UsdGeom, Gf
    head_h = max(0.006, dims["nut_h"] * 0.8)
    # head hex, centred at local origin; shank along local +Z (the long axis →
    # _part_long_axis_local picks local Z for the across-shaft bolt grasp).
    _define_hex_mesh(stage, path + "/head", dims["af"], head_h)
    shank = UsdGeom.Cylinder.Define(stage, path + "/shank")
    shank.CreateAxisAttr(UsdGeom.Tokens.z)
    shank.CreateRadiusAttr(dims["shank_d"] / 2.0)
    shank.CreateHeightAttr(shank_len)
    UsdGeom.XformCommonAPI(shank.GetPrim()).SetTranslate(
        Gf.Vec3d(0.0, 0.0, -(head_h / 2.0 + shank_len / 2.0)))


def _build_washer(stage, path: str, dims: dict) -> None:
    _define_annulus_mesh(stage, path + "/ring", dims["w_od"], dims["w_id"], dims["w_t"])


def spawn_part(stage, path: str, spec: PartSpec, table_top_z: float,
               rng: Optional[random.Random] = None) -> str:
    """Author a procedural part at `path`, pose it on the platform, give it a
    shape-accurate collider + realistic steel mass, and return the prim path.

    Author geometry → seat the bbox-min 1 mm above the platform top (no drop) →
    RigidBody + geometry-derived Mass → a compound collider matching the real shape
    (convex-hull hex/ring + capsule shank) bound to the grip material → steel look.
    The seat-don't-drop trick is from adapter.py::_spawn_nutsbolts (2620).
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf
    rng = rng or random
    dims = _MSIZE.get(spec.size, _MSIZE[DEFAULT_SIZE])

    prim = stage.DefinePrim(path, "Xform")
    shank_len = 0.0
    if spec.kind == "bolt":
        shank_len = rng.uniform(0.030, 0.050)
        _build_bolt(stage, path, dims, shank_len)
    elif spec.kind == "washer":
        _build_washer(stage, path, dims)
    else:
        _build_nut(stage, path, dims)

    # orientation: pose-class rotation about local X + yaw about world Z
    if spec.pose == "random":
        rotx = rng.choice([0.0, 90.0, 180.0])
    else:
        rotx = _POSE_ROTX.get((spec.kind, spec.pose), 0.0)

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    top_op = xf.AddTranslateOp()
    prov_z = table_top_z + 0.05                       # provisional, above the slab
    top_op.Set(Gf.Vec3d(spec.xy[0], spec.xy[1], prov_z))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(rotx, 0.0, spec.rotz_deg))

    # seat the part RESTING on the platform via its WORLD bbox (no drop → no
    # tunnel / contact-margin ejection). adapter.py:2620.
    try:
        wmin_z = float(UsdGeom.BBoxCache(
            0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        ).ComputeWorldBound(prim).ComputeAlignedBox().GetMin()[2])
        z = prov_z + (table_top_z + 0.001 - wmin_z)
    except Exception:
        z = table_top_z + 0.01
    top_op.Set(Gf.Vec3d(spec.xy[0], spec.xy[1], z))

    # ── collider that matches the REAL part shape ────────────────────────────
    # Convex-hull collider on the hex head / nut body / washer ring (the washer
    # hull fills the bore — correct for resting + rim grasping; nothing threads
    # through washers in this sim) + a CAPSULE on the bolt shank, so parts settle
    # and grasp like the real workshop components. PhysX builds a compound
    # collider from the multiple collision shapes under this one rigid body.
    grip = ensure_grasp_physics_material(stage)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(
        _part_mass(spec.kind, dims, shank_len))

    def _setup_collider(cprim):
        UsdPhysics.CollisionAPI.Apply(cprim)
        try:
            from pxr import PhysxSchema
            pc = PhysxSchema.PhysxCollisionAPI.Apply(cprim)
            pc.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
            pc.CreateRestOffsetAttr().Set(REST_OFFSET)
        except ImportError:
            pass
        if grip is not None:
            UsdShade.MaterialBindingAPI.Apply(cprim).Bind(
                grip, UsdShade.Tokens.weakerThanDescendants, "physics")

    def _hull_collider(mesh_path):
        mp = stage.GetPrimAtPath(mesh_path)
        if mp.IsValid():
            UsdPhysics.MeshCollisionAPI.Apply(mp).CreateApproximationAttr().Set("convexHull")
            _setup_collider(mp)

    if spec.kind == "bolt":
        head_h = max(0.006, dims["nut_h"] * 0.8)
        _hull_collider(path + "/head")
        r = dims["shank_d"] / 2.0
        cap = UsdGeom.Capsule.Define(stage, path + "/col_shank")
        cap.CreateAxisAttr(UsdGeom.Tokens.z)
        cap.CreateRadiusAttr(r)
        cap.CreateHeightAttr(max(shank_len - 2.0 * r, 0.004))   # total length ≈ shank_len
        UsdGeom.XformCommonAPI(cap.GetPrim()).SetTranslate(
            Gf.Vec3d(0.0, 0.0, -(head_h / 2.0 + shank_len / 2.0)))
        UsdGeom.Imageable(cap.GetPrim()).MakeInvisible()
        _setup_collider(cap.GetPrim())
    elif spec.kind == "washer":
        _hull_collider(path + "/ring")
    else:
        _hull_collider(path + "/body")

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        ensure_steel_material(stage), UsdShade.Tokens.strongerThanDescendants)
    return path


def repose_part(stage, path: str, spec: PartSpec, table_top_z: float,
                rng: Optional[random.Random] = None) -> str:
    """Re-pose an ALREADY-SPAWNED part (same kind/size) to a fresh spec instead
    of RemovePrim+respawn — prim churn is the suspected driver of the native
    heap drift after ~100 attempts (HANDOFF §9). Reuses spawn_part's pose-class
    rotation + seat-on-bbox placement and zeroes the body's velocities.

    Geometry is NOT rebuilt: a re-posed bolt keeps its spawned shank length
    (shank is random per spawn; across many cells/rounds diversity is fine)."""
    from pxr import UsdGeom, UsdPhysics, Gf
    rng = rng or random
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise ValueError(f"repose_part: no prim at {path}")

    if spec.pose == "random":
        rotx = rng.choice([0.0, 90.0, 180.0])
    else:
        rotx = _POSE_ROTX.get((spec.kind, spec.pose), 0.0)

    # Re-author the op stack exactly like spawn_part: after any simulation
    # steps PhysX has REPLACED the authored translate+rotateXYZ with its own
    # translate+orient writeback ops, so looking the originals up KeyErrors.
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    top_op = xf.AddTranslateOp()
    rot_op = xf.AddRotateXYZOp()
    prov_z = table_top_z + 0.05
    top_op.Set(Gf.Vec3d(spec.xy[0], spec.xy[1], prov_z))
    rot_op.Set(Gf.Vec3f(rotx, 0.0, spec.rotz_deg))
    try:
        wmin_z = float(UsdGeom.BBoxCache(
            0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        ).ComputeWorldBound(prim).ComputeAlignedBox().GetMin()[2])
        z = prov_z + (table_top_z + 0.001 - wmin_z)
    except Exception:
        z = table_top_z + 0.01
    top_op.Set(Gf.Vec3d(spec.xy[0], spec.xy[1], z))

    rb = UsdPhysics.RigidBodyAPI(prim)
    rb.GetVelocityAttr().Set(Gf.Vec3f(0.0))
    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0))
    return path


def set_part_physics_enabled(stage, path: str, enabled: bool):
    """Park/unpark a pooled part: a disabled rigid body drops out of the
    dynamics (its collider goes static — park it below the ground plane so
    nothing can touch it)."""
    from pxr import UsdPhysics
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(enabled)


# ── batch placement (scatter on the platform, min-separation) ──────────────
def scatter_xy(centre: tuple, n: int, spread: float = 0.03, min_sep: float = 0.03,
               rng: Optional[random.Random] = None) -> list:
    """n non-overlapping (x, y) slots within ±spread of centre (adapter.py:2595)."""
    rng = rng or random
    cx, cy = centre
    out = []
    for _ in range(n):
        x, y = cx, cy
        for _ in range(24):
            rx = cx + rng.uniform(-spread, spread)
            ry = cy + rng.uniform(-spread, spread)
            x, y = rx, ry
            if all((rx - px) ** 2 + (ry - py) ** 2 >= min_sep ** 2 for px, py in out):
                break
        out.append((x, y))
    return out


def bundle_xy(centre: tuple, n: int, rng: Optional[random.Random] = None,
              sep: float = 0.018) -> list:
    """n TOUCHING-cluster slots — a 'bundle' of parts dumped together (clutter).
    Much tighter than scatter_xy (parts ~touch at ~18 mm pitch), the scene the
    drag-separate strategy exists for."""
    rng = rng or random
    cx, cy = centre
    out = []
    for i in range(n):
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = sep * math.sqrt(rng.uniform(0.0, 1.0)) * math.sqrt(max(1, n - 1))
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out
