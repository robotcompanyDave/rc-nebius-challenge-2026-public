"""
Parametric parallel-jaw gripper — built procedurally in USD and welded into the
arm's articulation.

We deliberately do NOT reuse the Robotiq 2F-85 USD (licensing + a 5-bar
mimic-joint linkage that traces an *arc* as it closes and is fiddly to drive
headless). A parallel-jaw gripper moves its pads in a straight line on two
prismatic joints, so:
  • the controller's fingertip arc-compensation collapses to a fixed grip-Z, and
  • every part of the gripper is original to this repo (clean for a public repo).

Topology (all rigid bodies in the UR articulation, rooted at the arm base):
    wrist_3_link ──FixedJoint──► gripper/base
                                    ├─PrismaticJoint(+X)─► gripper/finger_left
                                    └─PrismaticJoint(−X)─► gripper/finger_right

The tool/approach axis is the arm EE-local **+Y** (matches `kinematics.tool_offset
= (0, 0.2422, 0)`); fingers extend along +Y so the fingertips sit at +Y = REACH.
The jaw axis is local **X**; both fingers share one scalar opening `d` (metres
each pad travels off centre) — d=0 closed (gap≈0), d=TRAVEL open (gap≈2·TRAVEL).

⚠ MOUNT FRAME: the exact rotation of the UR `wrist_3_link` USD frame vs the
kinematics convention must be confirmed on a GPU box (Step 0). `MOUNT_ROT_XYZ`
is the knob — start at identity, rotate if the gripper points the wrong way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# NOTE: `pxr` is imported lazily inside each function below. In Isaac's python the
# USD libraries are not importable until a SimulationApp has been constructed, and
# this module is imported (via robot.py) before that. See the boot-order note in
# sim_env.py. Every function here runs only after the app exists.


@dataclass
class GripperParams:
    reach: float = 0.2422          # kinematics tool_offset: wrist_3 → TCP along EE +Y (m).
                                   # The arm USD and kinematics.py both derive from the same
                                   # UR URDF, so the gripper fingertip authored at `reach`
                                   # from the wrist_3_link prim coincides with the kinematics
                                   # TCP — no structural offset.
    mount_offset: float = 0.0      # residual wrist_3_link → kinematics-tool gap along EE +Y
                                   # (m). Default 0: the ~0.1 m "gap" first seen on the GPU
                                   # box was the arm LAGGING the commanded descend, not a
                                   # geometric offset — it is handled by the grasp settle
                                   # (controller.GRASP_SETTLE), not here. Knob kept for any
                                   # real mount offset on a different arm USD.
    travel: float = 0.0425         # per-pad off-centre travel (m); 2·travel ≈ 85 mm stroke
    finger_len: float = 0.045      # pad length along +Y (m)
    finger_w: float = 0.022        # pad width along Z (m)
    pad_thick: float = 0.010       # pad thickness along the jaw axis X (m)
    base_size: float = 0.060       # gripper base cube edge (m)
    pad_friction: float = 1.4      # high-friction pads (match part grip material)
    drive_force: float = 40.0      # max finger drive force (N) — adapter UR10E_FMAX
    drive_stiffness: float = 1.0e5
    drive_damping: float = 1.0e3
    mount_rot_xyz: tuple = (0.0, 0.0, 0.0)   # deg; tune on GPU box if mis-oriented

    # Articulation DOF names the controller looks up.
    left_joint: str = "finger_left_joint"
    right_joint: str = "finger_right_joint"

    @property
    def open_val(self) -> float:   # prismatic position when OPEN (apart)
        return self.travel

    @property
    def close_val(self) -> float:  # prismatic position when CLOSED (together)
        return 0.0

    @property
    def stroke_mm(self) -> float:
        return 2.0 * self.travel * 1000.0


def _pad_material(stage, friction: float) -> "UsdShade.Material | None":
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


def _rigid_box(stage, path, size_xyz, translate, mass, color=(0.25, 0.27, 0.30),
               phys_mat=None):
    """A box rigid body authored the SAME way parts.py does: RigidBody/Mass on an
    UNSCALED Xform, with the box scale carried by a child Cube collider.

    Putting RigidBodyAPI directly on a *scaled* prim corrupts physics — PhysX bakes
    the prim scale into the body frame, so any joint LocalPos expressed in that
    frame is multiplied by the scale (a non-uniform scale collapsed the gripper
    along its length and mis-anchored the prismatic fingers). The body Xform here
    is unscaled, so joint local frames stay in true metres."""
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf
    body = UsdGeom.Xform.Define(stage, path)
    UsdGeom.XformCommonAPI(body.GetPrim()).SetTranslate(Gf.Vec3d(*translate))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr().Set(mass)

    geo = UsdGeom.Cube.Define(stage, path + "/geo")
    geo.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(Gf.Vec3f(*size_xyz))
    geo.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
    if phys_mat is not None:
        UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    return body.GetPrim()


def build_gripper(stage, tool_link_path: str, p: GripperParams) -> dict:
    """Author the gripper under `tool_link_path` and weld it to the articulation.

    Returns a dict describing the gripper (config the controller/robot consume):
      {gripper_path, open_val, close_val, left_joint, right_joint, reach, stroke_mm}
    """
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    # own mount path — "/gripper" is where the platform arm USD bakes the Robotiq
    # (deactivated by robot.load_arm when this parametric rung is selected)
    gpath = tool_link_path + "/gripper_pj"
    g = stage.DefinePrim(gpath, "Xform")
    if p.mount_rot_xyz != (0.0, 0.0, 0.0):
        UsdGeom.XformCommonAPI(g).SetRotate(Gf.Vec3f(*p.mount_rot_xyz))

    pad_mat = _pad_material(stage, p.pad_friction)

    # ── base / stem, fixed to the wrist flange ──────────────────────────────
    # The kinematics tool tip is `reach` (== tool_offset) out along EE +Y, so the
    # JAWS must grip there. We place the pad FAR END (fingertip) at Y = reach and
    # let the pads extend BACK toward the wrist; a stem bridges wrist→pads. This
    # is load-bearing: previously the pads sat at ~0.08 m while the controller
    # drove the TCP to 0.2422 m, so the jaws closed ~14 cm short (in the air).
    # Fingertip-at-reach (vs pad-centre-at-reach) also keeps the pads from
    # plunging through the table when the TCP descends to a flat part.
    half_pt = p.pad_thick / 2.0
    tip_from_mount = p.reach + p.mount_offset       # fingertip from the USD mount frame =
                                                    # kinematics TCP (reach from the flange)
    pad_cy = tip_from_mount - p.finger_len / 2.0    # pad centre; far end (tip) = TCP
    stem_front = pad_cy - p.finger_len / 2.0        # stem ends where the pads begin
    base_cy = stem_front / 2.0
    stem_cs = min(p.base_size, 0.04)                # stem cross-section (X, Z)
    base_prim = _rigid_box(stage, gpath + "/base",
                           (stem_cs, max(stem_front, 0.01), stem_cs),
                           (0.0, base_cy, 0.0), mass=0.30)

    fixed = UsdPhysics.FixedJoint.Define(stage, gpath + "/mount_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path(tool_link_path)])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(base_prim.GetPath())])
    # Local frames are LOAD-BEARING: a FixedJoint with no local poses defaults both
    # frames to (0,0,0) and welds the two body ORIGINS together — which yanks the
    # base (authored base_cy below the wrist) UP onto the wrist origin, lifting the
    # whole gripper by base_cy (~0.1 m) so the jaws close that far above the TCP.
    # Anchor the joint at the base's authored pose instead. (mount_rot_xyz is 0 by
    # default; a non-zero mount would also need LocalRot0 = R(mount_rot).)
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0.0, base_cy, 0.0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    # ── two finger pads on prismatic joints (jaw axis = X) ──────────────────
    # At d=0 the inner faces meet at x=0 (closed). Pad geometry is authored at the
    # closed rest pose; the joint translates it outward by d. The joint frame is
    # anchored at the pad's closed centre — on the BASE body it must be expressed
    # in that body's local frame (origin at base_cy), so subtract base_cy.
    for sign, jname, suffix in ((+1.0, p.left_joint, "left"),
                                (-1.0, p.right_joint, "right")):
        fpath = gpath + f"/finger_{suffix}"
        fprim = _rigid_box(stage, fpath, (p.pad_thick, p.finger_len, p.finger_w),
                           (sign * half_pt, pad_cy, 0.0), mass=0.05,
                           color=(0.15, 0.16, 0.18), phys_mat=pad_mat)

        joint = UsdPhysics.PrismaticJoint.Define(stage, gpath + f"/{jname}")
        joint.CreateAxisAttr(UsdPhysics.Tokens.x)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(base_prim.GetPath())])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(fprim.GetPath())])
        # We model both fingers with a +X axis but mirror the right one via a 180°
        # local-frame flip, so a single positive scalar `d` opens both symmetrically.
        joint.CreateLocalPos0Attr(Gf.Vec3f(sign * half_pt, pad_cy - base_cy, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        # axis points outward for each finger so positive drive = open
        if sign < 0:
            joint.CreateLocalRot1Attr(Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0)))  # 180° about Z → −X
            joint.CreateLocalRot0Attr(Gf.Quatf(0.0, Gf.Vec3f(0.0, 0.0, 1.0)))
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(p.travel + 0.002)

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(p.drive_force)
        drive.CreateTargetPositionAttr(p.open_val)     # start open
        drive.CreateStiffnessAttr(p.drive_stiffness)
        drive.CreateDampingAttr(p.drive_damping)

    return dict(
        kind="parametric",
        gripper_path=gpath,
        open_val=p.open_val,
        close_val=p.close_val,
        left_joint=p.left_joint,
        right_joint=p.right_joint,
        drive_joints=[p.left_joint, p.right_joint],
        reach=p.reach,
        stroke_mm=p.stroke_mm,
    )
