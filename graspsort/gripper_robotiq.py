"""
Real Robotiq 2F-85 gripper — the platform's actual gripper, for sim-to-real-
credible training (replaces the parametric parallel-jaw as the DEFAULT rung;
`gripper.py` remains selectable via GS_GRIPPER=parametric).

The arm USD copied from the platform (assets/virtual/imports/ur10/ur10.usd)
already has the 2F-85 BAKED IN at `<wrist_3_link>/gripper` — mounted by
rc-remote-platform/targets/ur10e/tools/add_gripper.py with the load-bearing
details handled there: nested ArticulationRootAPI stripped, an explicit
FixedJoint (localPos0 +0.0922 along wrist_3 +Y, localRot0 rotX(−90)) pinning
base_link to the flange, mimic joints in the payloads. Its `finger_joint`
composes into the UR articulation as one extra DOF.

So this module does NOT author the gripper. It:
  1. finds the baked gripper + its `finger_joint`,
  2. applies the FINGERNAIL grip tips (the workshop's dominant reliability
     lever: flat-M12-nut pick ~29% → 100%; UR10E_GRASP_RELIABILITY.md) — a
     grip pad run to a blunt bottom at each inner-finger tip, replacing the
     stock pad that sits ~12 mm above the tip,
  3. binds high-friction material to the finger colliders + raises the
     finger drive MaxForce to FMAX=40 N (more force does NOT help — measured),
  4. exposes the measured CLOSING-ARC model (sites/workshop/arc_profile.csv):
     the fingertip drops up to ~13.7 mm as the jaws close, bottoming at ~31 mm
     opening — the predictive z-comp the controller needs so a grip at width W
     lands ON the surface instead of ramming (open_tip_z = surface + clearance
     + arc_drop(W)).

Convention: openness 0 = OPEN, 1 = CLOSED (UR/platform convention);
finger_joint 0 rad open → ~0.8168 rad (47°) closed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ── measured 2F-85 (fingernail) closing arc — sites/workshop/arc_profile.csv ──
# columns: commanded openness, tip centre-to-centre opening (m), fingertip drop (m)
_ARC = np.array([
    # openness, opening_m, drop_m
    [0.00, 0.1027, 0.0000],
    [0.25, 0.0835, 0.0067],
    [0.50, 0.0619, 0.0112],
    [0.67, 0.0467, 0.0130],
    [0.83, 0.0312, 0.0137],
    [1.00, 0.0156, 0.0134],
])


def arc_drop_at_width(width_m: float) -> float:
    """Fingertip drop (m) when the jaws are at tip-opening `width_m` — how far the
    tips will be BELOW their open height by the time they grip a part that wide.
    Clamped to the measured range (≥ full-close width → the deep end)."""
    openings = _ARC[::-1, 1]          # ascending opening
    drops = _ARC[::-1, 2]
    return float(np.interp(width_m, openings, drops))


def fingertip_drop(openness: float) -> float:
    """Fingertip drop (m) at a commanded openness [0..1]."""
    return float(np.interp(openness, _ARC[:, 0], _ARC[:, 2]))


def openness_at_width(width_m: float) -> float:
    """Commanded openness that puts the tip opening at `width_m` (for stall gates)."""
    openings = _ARC[::-1, 1]
    op = _ARC[::-1, 0]
    return float(np.interp(width_m, openings, op))


ARC_DROP_MAX = float(_ARC[:, 2].max())          # ~13.7 mm


@dataclass
class RobotiqParams:
    joint_name: str = "finger_joint"
    open_val: float = 0.0                        # rad
    close_val: float = float(np.radians(47.0))   # ~0.8168 rad, from the platform map
    reach: float = 0.2422       # wrist_3 → TCP along EE +Y (0.0922 flange + ~0.15 gripper)
    fmax: float = float(os.environ.get("GS_FMAX", "40"))       # N; 40 measured best
    pad_friction: float = float(os.environ.get("GS_PAD_FRICTION", "1.4"))
    tip_len: float = float(os.environ.get("GS_TIPLEN", "0.016"))    # grip-face height (m)
    tip_thick_frac: float = float(os.environ.get("GS_TIPTHICK", "0.5"))
    # finger_joint drive authored in USD (see _bind_friction_and_force); force is
    # still capped by fmax, so these govern tracking speed, not clamp strength
    drive_stiffness: float = float(os.environ.get("GS_DRIVE_K", "1.0e4"))
    drive_damping: float = float(os.environ.get("GS_DRIVE_D", "1.0e2"))
    tips_on: bool = os.environ.get("GS_TIPS", "1") not in ("0", "", "false")


def _grip_material(stage, friction: float):
    """High-friction PhysX material for the finger pads (grasp hold by friction)."""
    from pxr import UsdShade, UsdPhysics
    try:
        from pxr import PhysxSchema
    except ImportError:
        return None
    path = "/World/Looks/JawPad"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        mat = UsdShade.Material.Define(stage, path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(friction)
        pm.CreateDynamicFrictionAttr().Set(friction)
        pm.CreateRestitutionAttr().Set(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        prim = mat.GetPrim()
    return UsdShade.Material(prim)


def _apply_fingernail_tips(stage, tool_link_path: str, p: RobotiqParams) -> int:
    """Port of adapter.py::_apply_fingernail_tips (same geometry, same values).

    2F-85 inner_finger LOCAL frame: the finger extends along +Z so the TIP = max
    local-Z; the grip face is the inner Y side (left_inner_finger toward +Y,
    right toward −Y). The tip box is a CHILD of the finger so it inherits the
    finger's world pose. Stock colliders under each finger are disabled — the
    tip box becomes the grip surface."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    mat = _grip_material(stage, p.pad_friction)
    root = stage.GetPrimAtPath(tool_link_path)
    if not root or not root.IsValid():
        return 0
    want = {"left_inner_finger": +1.0, "right_inner_finger": -1.0}
    found = {fn: None for fn in want}
    for d in Usd.PrimRange(root):
        nm = d.GetName()
        if nm in want and found[nm] is None:
            found[nm] = d
    done = 0
    for fname, inner_sign in want.items():
        fprim = found[fname]
        if fprim is None or not fprim.IsValid():
            print(f"[robotiq] fingernail: {fname} NOT FOUND", flush=True)
            continue
        lb = bc.ComputeUntransformedBound(fprim).ComputeAlignedBox()
        lmn, lmx = lb.GetMin(), lb.GetMax()
        zt = lmx[2]
        zb = zt - p.tip_len
        inner_y = lmx[1] if inner_sign > 0 else lmn[1]
        wY = (lmx[1] - lmn[1]) * p.tip_thick_frac
        bpath = f"{fprim.GetPath().pathString}/fingernail_tip"
        cube = UsdGeom.Cube.Define(stage, bpath)
        cube.GetSizeAttr().Set(1.0)
        capi = UsdGeom.XformCommonAPI(cube.GetPrim())
        capi.SetTranslate(Gf.Vec3d((lmn[0] + lmx[0]) / 2.0,
                                   inner_y - inner_sign * (wY / 2.0),
                                   (zb + zt) / 2.0))
        capi.SetScale(Gf.Vec3f(float(lmx[0] - lmn[0]), float(wY), float(p.tip_len)))
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.42, 0.45, 0.50)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if PhysxSchema is not None:
            pc = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            pc.CreateContactOffsetAttr().Set(0.002)
            pc.CreateRestOffsetAttr().Set(0.0)
        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        n_off = 0
        for d in Usd.PrimRange(fprim):
            if d == fprim or "fingernail_tip" in d.GetPath().pathString:
                continue
            if d.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(d).CreateCollisionEnabledAttr().Set(False)
                n_off += 1
        done += 1
    return done


def _bind_friction_and_force(stage, tool_link_path: str, p: RobotiqParams) -> tuple:
    """Port of adapter.py::_ensure_gripper_friction: high-friction material on every
    finger collider + MaxForce on the actuated finger_joint drive.

    PLUS (this env only): author the finger drive STIFFNESS/DAMPING in USD. The
    asset ships stiffness=3.0, which produces ~no torque here — measured: command
    full close, the joint stalls at 1.6% with 0.025 N·m effort while the
    articulation-view set_gains(1e5) echoes back but never reaches this joint's
    PhysX drive. The parametric jaw never hit this because its prismatic drives
    are USD-authored strong (gripper.py drive_stiffness=1e5). MaxForce still caps
    the clamp at FMAX, so grasp force stays the measured 40 N."""
    from pxr import Usd, UsdShade, UsdPhysics
    mat = _grip_material(stage, p.pad_friction)
    link = stage.GetPrimAtPath(tool_link_path)
    if not link.IsValid():
        return (0, 0)
    n_bound = n_drive = 0
    for sub in Usd.PrimRange(link):
        if mat is not None and sub.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(sub).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            n_bound += 1
        if sub.IsA(UsdPhysics.RevoluteJoint) and p.joint_name in sub.GetName():
            drive = UsdPhysics.DriveAPI.Apply(sub.GetPrim(), "angular")
            drive.CreateMaxForceAttr().Set(p.fmax)
            if sub.GetName() == p.joint_name:      # the ACTUATED joint only
                drive.CreateStiffnessAttr().Set(p.drive_stiffness)
                drive.CreateDampingAttr().Set(p.drive_damping)
                drive.CreateTypeAttr("force")
            n_drive += 1
    return (n_bound, n_drive)


def finger_prims(stage, tool_link_path: str) -> list:
    """The two inner-finger prims — the soft-surface dent feet + the bodies to
    collision-filter from the platform (adapter.py::_finger_prims)."""
    from pxr import Usd
    prims = []
    root = stage.GetPrimAtPath(tool_link_path)
    if root and root.IsValid():
        want = {"left_inner_finger", "right_inner_finger"}
        for d in Usd.PrimRange(root):
            if d.GetName() in want:
                prims.append(d)
    return prims


def gripper_body_prims(stage, tool_link_path: str) -> list:
    """All rigid-body links under the gripper — what the soft surface filters from
    the platform colliders (the 2F-85 colliders are instanced, so filtering is by
    rigid body; adapter.py::_ensure_soft_surface)."""
    from pxr import Usd, UsdPhysics
    out = []
    root = stage.GetPrimAtPath(tool_link_path + "/gripper")
    if not (root and root.IsValid()):
        root = stage.GetPrimAtPath(tool_link_path)
    if root and root.IsValid():
        for d in Usd.PrimRange(root):
            if d.HasAPI(UsdPhysics.RigidBodyAPI):
                out.append(d)
    return out


def setup_robotiq(stage, tool_link_path: str,
                  p: RobotiqParams | None = None) -> dict:
    """Find the baked-in 2F-85 under `tool_link_path`, apply fingernails +
    friction + FMAX, and return the gconf the robot layer consumes."""
    p = p or RobotiqParams()
    gpath = tool_link_path + "/gripper"
    g = stage.GetPrimAtPath(gpath)
    if not g or not g.IsValid():
        raise RuntimeError(
            f"Robotiq gripper not found at {gpath} — use the gripper-baked arm USD "
            f"(assets/virtual/imports/ur10/ur10.usd, copied from the platform) or "
            f"set GS_GRIPPER=parametric")
    # Physics variant: the asset ships None | Physics | Physx_Mimic | Physx_Loop.
    variant = os.environ.get("GS_ROBOTIQ_VARIANT", "").strip()
    if variant:
        vs = g.GetVariantSet("Physics")
        if vs and vs.IsValid():
            ok = vs.SetVariantSelection(variant)
            print(f"[robotiq] Physics variant -> {variant} (ok={ok})", flush=True)
    # Mimic gearing flip (diagnostic knob): at the observed lock the LOOP geometry
    # pulls right_outer_knuckle SAME-sign as finger_joint (+1.2 ratio) while the
    # authored gearing −1 demands opposite — the two constraints fight and freeze
    # the linkage. GS_MIMIC_FLIP=1 flips the −1 gearings to +1.
    if os.environ.get("GS_MIMIC_FLIP", "0") not in ("0", "", "false"):
        from pxr import Usd
        n_flip = 0
        for sub in Usd.PrimRange(g):
            for attr in sub.GetAttributes():
                nm = attr.GetName()
                if nm.startswith("physxMimicJoint:") and nm.endswith(":gearing"):
                    v = attr.Get()
                    if v is not None and float(v) < 0:
                        attr.Set(-float(v))
                        n_flip += 1
        print(f"[robotiq] mimic gearing flipped on {n_flip} joint(s)", flush=True)
    # Self-collisions OFF on the owning articulation root: the 2F-85's linkage
    # colliders overlap by design, so with self-collision on the mechanism JAMS
    # at ~1°. The asset's own articulation API said enabledSelfCollisions=0, but
    # add_gripper.py had to strip that API (nested articulation roots are
    # illegal), so the arm root must carry the setting instead.
    try:
        from pxr import Usd, UsdPhysics, PhysxSchema
        root = None
        prim = stage.GetPrimAtPath(tool_link_path)
        while prim and prim.IsValid():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                root = prim
                break
            prim = prim.GetParent()
        if root is None:
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    root = prim
                    break
        if root is not None:
            api = PhysxSchema.PhysxArticulationAPI.Apply(root)
            before = api.GetEnabledSelfCollisionsAttr().Get()
            api.CreateEnabledSelfCollisionsAttr().Set(False)
            # the Robotiq asset's own articulation API (stripped by add_gripper —
            # nested roots are illegal) carried solverPositionIterationCount=64;
            # the mimic/loop closures need those iterations to converge
            api.CreateSolverPositionIterationCountAttr().Set(64)
            api.CreateSolverVelocityIterationCountAttr().Set(4)
            print(f"[robotiq] articulation root {root.GetPath()}: "
                  f"selfCollisions {before} -> False, posIters=64", flush=True)
    except Exception as e:
        print(f"[robotiq] self-collision setup: {e}", flush=True)
    tips = _apply_fingernail_tips(stage, tool_link_path, p) if p.tips_on else 0
    n_bound, n_drive = _bind_friction_and_force(stage, tool_link_path, p)
    print(f"[robotiq] 2F-85 ready: fingernail tips={tips}, colliders bound={n_bound}, "
          f"drives forced={n_drive} (FMAX={p.fmax}N)", flush=True)
    return dict(
        kind="robotiq",
        gripper_path=gpath,
        drive_joints=[p.joint_name],     # single actuated DOF; mimics live in the USD
        open_val=p.open_val,
        close_val=p.close_val,
        reach=p.reach,
        stroke_mm=85.0,
        arc_drop_max=ARC_DROP_MAX,
    )
