#!/usr/bin/env python3
"""
PICK LAB — learn the two-finger press→drag→roll-up washer pick on the compliant
surface, across material variants, with MANY rigs in ONE physics scene.

The maneuver (diagrams data/2026-07-03/1029-press_design/01_pick_sequence_v2.svg):
  1. PRESS  finger A (low-friction) presses one side of the flat washer into the
            soft surface → the washer tilts, far edge rises out of the dish.
  2. BDOWN  finger B (high-friction, compliant) descends OUTSIDE the raised edge.
  3. DRAG   B drags inward along/through the surface to connect with the raised
            edge's SIDE.
  4. LIFT+CLOSE  both fingers rise while B keeps closing → the washer rolls up
            to vertical and ends pinched between the two fingers, off the surface.

"Training" = CEM (cross-entropy) optimization of the maneuver parameters θ,
one candidate per rig per round, all rigs stepping in the same scene at once.
Finger B is a dynamic body on a spring prismatic (the gripper.py construct), so
grip force is bounded/compliant — no kinematic crushing.

This scene is STANDALONE (no arm) — physics is proven identical to the full env;
rendering here is not (dynamic prims don't render in the minimal boot), so this
lab is numbers-only. Use tools/render_pick.py to replay a θ in the full env.

    docker/run.sh tools/pick_lab.py
Env:
  GS_PL_RIGS (6)      rigs (=CEM population) per round
  GS_PL_ROUNDS (8)    CEM rounds per material
  GS_PL_CELL (0.005)  tile size (5mm lab default; render uses 4mm)
  GS_PL_MATS          path to a JSON list of material configs (see DEFAULT_MATS)
  GS_PL_OUT           output dir (default dated)
"""
import datetime
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

RIGS = int(os.environ.get("GS_PL_RIGS", "6"))
SPAN = float(os.environ.get("GS_PL_SPAN", "0.060"))   # bed size per rig
ROUNDS = int(os.environ.get("GS_PL_ROUNDS", "8"))
CELL = float(os.environ.get("GS_PL_CELL", "0.005"))
_now = datetime.datetime.now()
OUT = os.environ.get("GS_PL_OUT", os.path.join(
    "data", _now.strftime("%Y-%m-%d"), _now.strftime("%H%M") + "-pick_lab"))

SURF_Z = 0.50
SPACING = 0.30
FW, FH = 0.008, 0.030          # finger cross-section / height
W_R = 0.012                    # m12 washer outer radius
W_T = 0.0025                   # m12 washer thickness

# default material ladder — M0 is the surface as validated yesterday
DEFAULT_MATS = [
    dict(name="M0_asis", stiffness=300.0, damping=8.0,
         couple=150.0, couple_damp=8.0, falloff=0.0, cutoff=0.10),
]

# θ bounds: [a_off, press_depth, b_out, b_tip, b_drag_end, gap_final, lift_h]
# b_tip reaches well below the surface: the coupled dish swallows the washer
# (it sinks ~flat), so B must PLOW through the foam to catch the rim's side.
# gap_final goes BELOW the 2.5mm washer thickness — B's compliant spring absorbs
# the overlap into a real pinch force (M0 finding: gap>=2.8mm never squeezes, the
# washer wedges and slips out at ~12mm of lift, every single round).
# Bounds widened after the neoprene ladder pinned press/a_off at HI and b_drag
# at LO: a wide dish resists one-sided tilt, so the optimum wants a DEEPER
# press further out and little-to-no drag (descend + close does the work).
TH_LO = np.array([0.004, 0.0010, 0.002, -0.007, 0.000, 0.0015, 0.020])
TH_HI = np.array([0.013, 0.0080, 0.010, 0.002, 0.014, 0.0080, 0.050])
TH_MU0 = np.array([0.007, 0.0032, 0.005, -0.003, 0.008, 0.0028, 0.035])

# phase windows (s): settle, press, b-down, drag, lift+close, hold
T_SET, T_PRS, T_BDN, T_DRG, T_LFT, T_HLD = 0.5, 1.2, 0.6, 1.0, 1.5, 0.7
T_TOTAL = T_SET + T_PRS + T_BDN + T_DRG + T_LFT + T_HLD
PHYS_DT = 1.0 / 240.0

# ── PARALLEL-GRIPPER mode (mat["mode"]=="parallel", 2026-07-04 late) ─────────
# Both fingers bound as ONE gripper: they descend together, the press side (A,
# ~50% over the rim) touches the part first; when the BLUE finger's spring
# deflects (= it touched the part's rising side — the same signal a real
# gripper reads as motor current), pull up and close to grip at the part's top.
# Success bar per David: clear the PRESS-DOWN height, not the start height.
# θp = [a_over, g0, press_depth, close_gain, gap_final, rise_h, brace]
#  a_over: fraction of A's width overhanging the rim (0.5 = half on, half off)
#  g0: finger centre-to-centre spacing at approach
#  close_pow: close progress = rise_progress**close_pow during the pull-up.
#  >1 = the close LAGS (washer rolls up on B's friction first, gap completes
#  at the end); <1 = the close leads. A linear lead jammed the still-diagonal
#  washer between the faces (grip_cem1: smooth 54 deg rolls, zero captures —
#  at 50 deg the washer spans 12mm+, far wider than the closing gap).
#  brace: small extra press right after the touch (B holds) — seats the edge
#  against B's face before the pull-up.
# g0 note: the rising far rim arcs UP and INWARD (rotation about the pressed
# edge), so B's inner face must start at/just inside the rim radius to be
# lifted into — face = g0 − (W_R − a_c offset) − FW/2 ≈ g0 − 16mm → rim at
# g0 ≈ 28mm. Bounds bracket that.
# a_over capped below the flingy 0.65 bound-pin, press below the pop regime —
# the smooth-ramp reward (violence < failure) does the rest.
# th 8-9 (2026-07-05, Stage T carry round): gap_carry = post-capture squeeze
# (splits "loose enough to roll" from "tight enough to hold"); pitch_deg =
# wrist pitch during carry (EE is on a 6-DOF arm — David approved pitch/yaw/
# roll experiments for the lift; pitch pivots about the FINGERTIPS)
THP_LO = np.array([0.35, 0.025, 0.0012, 0.4, 0.0015, 0.008, 0.0000,
                   0.0008, -20.0, -15.0])
THP_HI = np.array([0.62, 0.031, 0.0045, 4.0, 0.0030, 0.032, 0.0025,
                   0.0030, 20.0, 15.0])
# θ[9] = yaw_deg: EE yaw about the washer centre, ramping in EARLY CLOSE
# (from the grip moment, full by end of close) — a gentleness axis: a slight
# twist while pinching can seat the rim with less normal force. 0 = legacy.
THP_MU0 = np.array([0.52, 0.028, 0.0030, 1.8, 0.0021, 0.018, 0.0010,
                    0.0016, 0.0, 0.0])
T_DSC, T_WAIT, T_LC = 1.2, 1.2, 2.0       # descend / max wait / brace+pull-up
T_CARRY, CARRY_H = 1.5, 0.045             # carry phase: rise to 45mm hold
T_BRACE = 0.35                            # seat-the-edge sub-phase of T_LC
TRIG_DX = 0.00022                         # B deflection that counts as "touched"
TRIG_HOLD = 10                            # consecutive physics steps above threshold
# (0.8mm never fired across 128 evals, 0.35mm barely; the gentle edge-rise only
# brushes B. Lower threshold + debounce: fire on sustained contact, reject spikes.)

# ── jitter (robust-selection phase) ──────────────────────────────────────────
# During SEARCH each eval gets light noise so CEM can't camp on knife-edges;
# for SELECTION the top candidates face a fixed 8-jitter PANEL (one per rig)
# and the winner is the best WORST-2-MEAN. jitter = (washer dx, dy, gripper dz).
# Placement-deviation half-range (m) — the surface is not homogeneous, so the
# washer never lands exactly on-slot. Widened default ±1.5 mm (GS_PL_JIT_XY).
JIT_XY = float(os.environ.get("GS_PL_JIT_XY", "0.0015"))
# EE-roll half-range (deg): each rig gets a fixed lattice rotation spread over
# ±JIT_ROLL (see main), so the CEM population spans approach angles and the
# selected θ is robust to EE roll over the heterogeneous surface.
JIT_ROLL = float(os.environ.get("GS_PL_JIT_ROLL", "20.0"))
SEARCH_JIT = (JIT_XY, JIT_XY, 0.0003)     # uniform half-ranges (washer dx, dy, gripper dz)
PANEL = [(0.0, 0.0, 0.0),
         (+JIT_XY, 0.0, 0.0), (-JIT_XY, 0.0, 0.0),
         (0.0, +JIT_XY, 0.0), (0.0, -JIT_XY, 0.0),
         (0.0, 0.0, +0.0005), (0.0, 0.0, -0.0005),
         (+0.7 * JIT_XY, +0.7 * JIT_XY, -0.0005)]
GAUNT_K = int(os.environ.get("GS_PL_GAUNT_K", "4"))
T_TOTAL_P = T_SET + T_DSC + T_WAIT + T_LC + T_HLD


def _ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


class PickRig:
    """One foundation + washer + two fingers, offset to its own patch of world."""

    def __init__(self, stage, rig_id, centre, mat, roll_deg=0.0):
        self.stage = stage
        self.id = rig_id
        self.c = centre
        self.mat = mat
        self._roll_deg = float(roll_deg)   # fixed lattice rotation = EE-roll axis
        self.base = f"/World/Rig{rig_id}"
        self.found = None
        self.wpath = None
        self.theta = TH_MU0.copy()
        self._w_home = None
        self._leveling = bool(mat.get("leveling", False))
        self._ell = float(mat.get("ell", 0.006))
        self._lvl_ct = 0
        self._mode = mat.get("mode", "legacy")
        self._ph = 0            # parallel mode: 0 descend, 2 lift+close
        self._t3 = None         # time the grip trigger fired
        self._zpress = None     # A-bottom height at the grip moment
        self._triggered = False

    # ── build ────────────────────────────────────────────────────────────────
    def build(self):
        from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
        from graspsort.soft_foundation import SpringFoundation
        from graspsort import parts
        st = self.stage
        cx, cy = self.c
        UsdGeom.Xform.Define(st, self.base)

        m = self.mat
        self.found = SpringFoundation(
            st, self.c, SURF_Z, span=SPAN, cell=CELL,
            stiffness=m["stiffness"], damping=m["damping"],
            couple=m["couple"], couple_damp=m["couple_damp"],
            couple_falloff=m.get("falloff", 0.0),
            couple_cutoff=m.get("cutoff", 0.10),
            travel=m.get("travel", 0.014),   # thin pads can't compress 14mm
            rotz_deg=self._roll_deg,
            visible=False, parent=self.base + "/Foundation").build()

        spec = parts.PartSpec(kind="washer", size="m12", pose="flat",
                              xy=self.c, rotz_deg=0.0)
        self.wpath = parts.spawn_part(st, self.base + "/Washer", spec, SURF_Z)
        wprim = st.GetPrimAtPath(self.wpath)
        t = UsdGeom.Xformable(wprim).GetOrderedXformOps()[0].Get()
        self._w_home = (float(t[0]), float(t[1]), float(t[2]))

        # finger materials
        def phys_mat(path, fric):
            UsdShade.Material.Define(st, path)
            fm = UsdPhysics.MaterialAPI.Apply(st.GetPrimAtPath(path))
            fm.CreateStaticFrictionAttr(fric)
            fm.CreateDynamicFrictionAttr(fric)
            fm.CreateRestitutionAttr(0.0)
            return UsdShade.Material(st.GetPrimAtPath(path))
        matA = phys_mat(self.base + "/MatA", 0.10)   # press finger: slippery
        matB = phys_mat(self.base + "/MatB", 1.20)   # grip finger: grippy

        def kin_finger(path, x, mat_, shape="box"):
            xf = UsdGeom.Xform.Define(st, path)
            UsdGeom.XformCommonAPI(xf.GetPrim()).SetTranslate(
                Gf.Vec3d(x, cy, SURF_Z + 0.08))
            rb = UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            rb.CreateKinematicEnabledAttr(True)
            if shape == "capsule":
                # rounded fingertip: the hemispherical bottom lets B PLOW through
                # the foam when dragging (angled contact normals push tiles DOWN);
                # a box bottom corner jams against the laterally-rigid tile sides
                geo = UsdGeom.Capsule.Define(st, path + "/geo")
                geo.CreateAxisAttr(UsdGeom.Tokens.z)
                geo.CreateRadiusAttr(FW / 2.0)
                geo.CreateHeightAttr(FH - FW)
            else:
                geo = UsdGeom.Cube.Define(st, path + "/geo")
                geo.GetSizeAttr().Set(1.0)
                UsdGeom.XformCommonAPI(geo.GetPrim()).SetScale(
                    Gf.Vec3f(FW, FW, FH))
            UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
            pc = PhysxSchema.PhysxCollisionAPI.Apply(geo.GetPrim())
            pc.CreateContactOffsetAttr().Set(0.0015)
            pc.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(
                mat_, UsdShade.Tokens.weakerThanDescendants, "physics")
            return xf.GetPrim()

        # A: kinematic press finger
        self.primA = kin_finger(self.base + "/FingerA", cx - 0.007, matA)
        # B carrier: kinematic, NO collider
        carr = UsdGeom.Xform.Define(st, self.base + "/CarrierB")
        UsdGeom.XformCommonAPI(carr.GetPrim()).SetTranslate(
            Gf.Vec3d(cx + 0.020, cy, SURF_Z + 0.08))
        rbc = UsdPhysics.RigidBodyAPI.Apply(carr.GetPrim())
        rbc.CreateKinematicEnabledAttr(True)
        self.primC = carr.GetPrim()
        # B: dynamic grip finger, sprung to the carrier along X (compliant pinch:
        # normal force = drive spring, bounded by maxForce — no kinematic crush).
        # Tip shape from the material config: neoprene-with-leveling should work
        # with a SQUARE tip (fig/02); the capsule was the gel-era workaround.
        self.primB = kin_finger(self.base + "/FingerB", cx + 0.020, matB,
                                shape=self.mat.get("btip", "capsule"))
        UsdPhysics.RigidBodyAPI(self.primB).CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(self.primB).CreateMassAttr(0.040)
        jt = UsdPhysics.PrismaticJoint.Define(st, self.base + "/BSpring")
        jt.CreateAxisAttr(UsdPhysics.Tokens.x)
        jt.CreateBody0Rel().SetTargets([Sdf.Path(self.base + "/CarrierB")])
        jt.CreateBody1Rel().SetTargets([Sdf.Path(self.base + "/FingerB")])
        jt.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        jt.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        jt.CreateLowerLimitAttr(-0.020)
        jt.CreateUpperLimitAttr(0.020)
        drv = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "linear")
        drv.CreateTypeAttr("force")
        drv.CreateTargetPositionAttr(0.0)
        drv.CreateStiffnessAttr(8000.0)
        drv.CreateDampingAttr(90.0)
        drv.CreateMaxForceAttr(60.0)   # plowing the foam takes real push
        return self

    # ── control ──────────────────────────────────────────────────────────────
    def _set_pos(self, prim, x, z, pitch_deg=0.0, y=None, yaw_deg=0.0):
        from pxr import UsdGeom, Gf
        if pitch_deg:
            # pitch about the FINGERTIP (keep the grip point stationary):
            # tip = (x, z - FH/2); center = tip + (FH/2)*(sin p, cos p)
            rad = math.radians(pitch_deg)
            x = x + math.sin(rad) * FH / 2
            z = (z - FH / 2) + math.cos(rad) * FH / 2
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(x, self.c[1] if y is None else y, z))
        api.SetRotate(Gf.Vec3f(0.0, pitch_deg, yaw_deg))

    def _reauthor(self, prim, pos):
        """Teleport a dynamic body. Author translate + ORIENT — the exact op pair
        PhysX's writeback uses. (Re-authoring translate+rotateXYZ mid-session left
        rotation writeback DEAD: translation updated but the body's USD rotation
        froze at identity — washers 'never tilted'. orient keeps writeback alive.)"""
        from pxr import UsdGeom, UsdPhysics, Gf
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))

    def reset(self, theta, jitter=(0.0, 0.0, 0.0)):
        from pxr import Gf
        self.theta = np.asarray(theta, dtype=np.float64)
        self._jit = tuple(jitter)
        self._pitch = 0.0
        cx = self.c[0]
        st = self.stage
        wh = (self._w_home[0] + self._jit[0],
              self._w_home[1] + self._jit[1], self._w_home[2])
        self._reauthor(st.GetPrimAtPath(self.wpath), wh)
        hi = SURF_Z + 0.06 + FH / 2
        if self._mode == "parallel":
            a_over, g0 = self.theta[0], self.theta[1]
            a_c = cx - W_R - (a_over - 0.5) * FW   # 0.5 = half over the rim
            bx0 = a_c + g0
            self._ph = 0
            self._t3 = None
            self._zpress = None
            self._triggered = False
            self._trig_ct = 0
            self._vc = 0.0          # close fraction (eye-diagram x-axis)
        else:
            a_off, _pd, b_out, _bt, _bd, _gap, _lh = self.theta
            a_c, bx0 = cx - a_off, cx + W_R + b_out
        self._set_pos(self.primA, a_c, hi)
        self._set_pos(self.primC, bx0, hi)
        self._reauthor(st.GetPrimAtPath(self.base + "/FingerB"),
                       (bx0, self.c[1], hi))
        self.max_tilt = 0.0
        self.lift_tilt = 0.0
        self.flew = False
        self.max_dtilt = 0.0        # biggest tilt jump between samples (snap metric)
        self._prev_tilt = None
        self._lvl_ct = 0
        if self._leveling:
            self.found.level_targets([], self._ell)   # relax all targets

    def drive(self, t):
        """Kinematic waypoints for A and B-carrier at sim time t."""
        if self._mode == "parallel":
            return self._drive_parallel(t)
        cx = self.c[0]
        a_off, pd, b_out, b_tip, b_drag, gap, lift_h = self.theta
        ax = cx - a_off
        bx0 = cx + W_R + b_out
        a_hi = SURF_Z + 0.06 + FH / 2
        a_press = SURF_Z - pd + FH / 2
        b_work = SURF_Z + b_tip + FH / 2
        bx_close = (ax + FW / 2) + gap + FW / 2      # inner-face gap = gap

        t1 = T_SET
        t2 = t1 + T_PRS
        t3 = t2 + T_BDN
        t4 = t3 + T_DRG
        t5 = t4 + T_LFT
        if t < t1:                                    # settle
            az, bx, bz = a_hi, bx0, a_hi
        elif t < t2:                                  # press
            u = _ease((t - t1) / T_PRS)
            az, bx, bz = a_hi + (a_press - a_hi) * u, bx0, a_hi
        elif t < t3:                                  # B descends outside
            u = _ease((t - t2) / T_BDN)
            az, bx, bz = a_press, bx0, a_hi + (b_work - a_hi) * u
        elif t < t4:                                  # drag inward
            u = _ease((t - t3) / T_DRG)
            az, bx, bz = a_press, bx0 + (cx + b_drag - bx0) * u, b_work
        elif t < t5:                                  # close AT DEPTH, then lift
            # SVG order: grip the raised edge first, THEN rise. Closing while
            # rising leaves the rolled-up washer STANDING in the foam (center
            # +11.7mm = OD/2 - sink) as the still-open fingers sail past it.
            u = _ease((t - t4) / T_LFT)
            CLOSE_FRAC = 0.4
            if u < CLOSE_FRAC:                        # pinch at working height
                v = u / CLOSE_FRAC
                az = a_press
                bx = (cx + b_drag) + (bx_close - (cx + b_drag)) * v
                bz = b_work
            else:                                     # lift with the pinch held
                v = (u - CLOSE_FRAC) / (1.0 - CLOSE_FRAC)
                az = a_press + (SURF_Z + lift_h + FH / 2 - a_press) * v
                bx, bz = bx_close, b_work + (az - a_press)
        else:                                         # hold
            az = SURF_Z + lift_h + FH / 2
            bx, bz = bx_close, b_work + (az - a_press)
        self._set_pos(self.primA, ax, az)
        self._set_pos(self.primC, bx, bz)

        # proximity leveling (fig/02): conform tile targets to nearby bodies at
        # 60 Hz so drags meet ramps, not walls — enables the square tip.
        if self._leveling:
            self._lvl_ct += 1
            if self._lvl_ct % 4 == 1:
                bodies = [(ax, self.c[1], FW / 2, FW / 2, az - FH / 2),
                          (bx, self.c[1], FW / 2, FW / 2, bz - FH / 2)]
                wb = self._washer_footprint()
                if wb:
                    bodies.append(wb)
                self.found.level_targets(bodies, self._ell)

    def _drive_parallel(self, t):
        """The bound-finger gripper: descend together (A touches the part
        first), slow-press while WAITING for B's spring to feel the rising
        side, then pull up + close to grip near the part's top."""
        from pxr import UsdGeom
        cx = self.c[0]
        a_over, g0, pd, close_pow, gap, rise_h, brace = self.theta[:7]
        gap_carry = self.theta[7] if len(self.theta) > 7 else gap
        pitch_deg = self.theta[8] if len(self.theta) > 8 else 0.0
        a_c = cx - W_R - (a_over - 0.5) * FW
        dz = self._jit[2] if hasattr(self, "_jit") else 0.0   # height error
        hi = SURF_Z + 0.06 + FH / 2 + dz
        zp = SURF_Z - pd + FH / 2 + dz
        bx0 = a_c + g0
        bx_close = a_c + FW + gap                  # inner-face gap = gap
        t1, t2 = T_SET, T_SET + T_DSC
        t3max = t2 + T_WAIT

        if self._ph < 2:
            if t < t1:                             # settle
                z, bx = hi, bx0
            elif t < t2:                           # descend together
                u = _ease((t - t1) / T_DSC)
                z, bx = hi + (zp - hi) * u, bx0
            else:                                  # hold press, wait for touch
                z, bx = zp, bx0
                # B's spring deflection = the touch sensor (what a real gripper
                # reads as finger motor current)
                xb = float(UsdGeom.XformCache().GetLocalToWorldTransform(
                    self.stage.GetPrimAtPath(self.base + "/FingerB"))
                    .ExtractTranslation()[0])
                if abs(xb - bx0) > TRIG_DX:
                    self._trig_ct += 1
                    if self._trig_ct >= TRIG_HOLD:   # sustained contact, not a spike
                        self._triggered = True
                else:
                    self._trig_ct = 0
                if self._triggered or t >= t3max:  # grip moment (or timeout)
                    self._ph = 2
                    self._t3 = t
                    self._zpress = z
        if self._ph >= 2:
            # touch -> tiny BRACE (seat the edge on B's face) -> PULL UP AND
            # TOGETHER: rise and close as one motion, with close_gain deciding
            # how much the close leads. B rising with the edge held by friction
            # is what rolls the washer up (the SVG mechanism).
            dt3 = t - self._t3
            if dt3 < T_BRACE:                      # seat the edge
                v = _ease(dt3 / T_BRACE)
                z = self._zpress - brace * v
                bx = bx0
            elif dt3 < T_LC:                       # pull up + together
                v = _ease(min((dt3 - T_BRACE) / (T_LC - T_BRACE), 1.0))
                vc = v ** close_pow                # >1: close lags, ends closed
                self._vc = vc
                z = self._zpress - brace + rise_h * v
                bx = bx0 + (bx_close - bx0) * vc
            else:                                  # CARRY: squeeze+pitch+rise
                w = _ease(min((dt3 - T_LC) / T_CARRY, 1.0))
                z_top = self._zpress - brace + rise_h
                z_car = SURF_Z + CARRY_H + FH / 2
                z = z_top + (z_car - z_top) * w
                bx = (a_c + FW + gap) + (gap_carry - gap) * w
                self._pitch = pitch_deg * w
                self._vc = 1.0
        pch = getattr(self, "_pitch", 0.0)
        # θ[9]: EE yaw about the washer centre, ramping through the close
        # ("early in the close manoeuvre") — 0 before the grip moment.
        yaw_p = float(self.theta[9]) if len(self.theta) > 9 else 0.0
        yw = 0.0
        if yaw_p and self._ph >= 2 and self._t3 is not None:
            yw = yaw_p * _ease(min((t - self._t3) / T_LC, 1.0))
        cy = self.c[1]
        if yw:
            cr, sr = math.cos(math.radians(yw)), math.sin(math.radians(yw))
            ax_r = cx + (a_c - cx) * cr
            ay_r = cy + (a_c - cx) * sr
            bx_r = cx + (bx - cx) * cr
            by_r = cy + (bx - cx) * sr
        else:
            ax_r, ay_r, bx_r, by_r = a_c, cy, bx, cy
        self._set_pos(self.primA, ax_r, z, pch, y=ay_r, yaw_deg=yw)
        self._set_pos(self.primC, bx_r, z, pch, y=by_r, yaw_deg=yw)

        if self._leveling:
            self._lvl_ct += 1
            if self._lvl_ct % 4 == 1:
                bodies = [(ax_r, ay_r, FW / 2, FW / 2, z - FH / 2),
                          (bx_r, by_r, FW / 2, FW / 2, z - FH / 2)]
                wb = self._washer_footprint()
                if wb:
                    bodies.append(wb)
                self.found.level_targets(bodies, self._ell)

    def _washer_footprint(self):
        from pxr import UsdGeom
        bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(
            self.stage.GetPrimAtPath(self.wpath)).ComputeAlignedBox()
        mn, mx = bb.GetMin(), bb.GetMax()
        if mx[0] <= mn[0]:
            return None
        return (float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2),
                float((mx[0] - mn[0]) / 2), float((mx[1] - mn[1]) / 2),
                float(mn[2]))

    # ── measurement ──────────────────────────────────────────────────────────
    def washer_state(self):
        from pxr import UsdGeom
        M = UsdGeom.XformCache().GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.wpath))
        tr = M.ExtractTranslation()
        n = M.TransformDir((0.0, 0.0, 1.0))
        ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
        tilt = math.degrees(math.acos(min(1.0, abs(n[2]) / ln)))
        return (float(tr[0]), float(tr[1]), float(tr[2])), tilt

    def observe(self, t=0.0):
        (wx, wy, wz), tilt = self.washer_state()
        cx, cy = self.c
        if abs(wx - cx) > 0.06 or abs(wy - cy) > 0.06:
            self.flew = True
        self.max_tilt = max(self.max_tilt, min(tilt, 90.0))
        # ARRIVAL-snap metric: biggest tilt jump between 20 Hz samples BEFORE
        # the grip phase only. The close deliberately rotates the washer to
        # vertical between the fingers (that rotation is the roll-up, not a
        # defect) — counting it flagged every neoprene success as a "snap".
        in_window = (self._ph < 2 if self._mode == "parallel"
                     else t < T_SET + T_PRS + T_BDN + T_DRG)
        if in_window:
            if self._prev_tilt is not None:
                self.max_dtilt = max(self.max_dtilt, abs(tilt - self._prev_tilt))
            self._prev_tilt = tilt
        # tilt sustained into the lift+close window — the roll-up signal (a drag
        # that flattens the washer scores 0 here even if the press tilted it)
        if t > T_SET + T_PRS + T_BDN + T_DRG:
            self.lift_tilt = max(getattr(self, "lift_tilt", 0.0), min(tilt, 90.0))
        return tilt

    def final_score(self):
        from pxr import UsdGeom
        (wx, wy, wz), tilt = self.washer_state()
        cx, cy = self.c
        a_off, _pd, _bo, _bt, _bd, gap, lift_h = self.theta[:7]
        xc = UsdGeom.XformCache()
        ax = float(xc.GetLocalToWorldTransform(
            self.primA).ExtractTranslation()[0])
        bx = float(xc.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.base + "/FingerB"))
            .ExtractTranslation()[0])
        inner_lo, inner_hi = ax + FW / 2 - 0.002, bx - FW / 2 + 0.002
        if self._mode == "parallel":
            # goal (David): clear the PRESS-DOWN height, not the start height —
            # the washer just has to hang free of the dish it was pressed into.
            wb = self._washer_footprint()
            floor = ((self._zpress - self.theta[6] - FH / 2)
                     if self._zpress is not None
                     else SURF_Z - self.theta[2] - self.theta[6])
            clearance = (wb[4] - floor) if wb else 0.0
            captured = (tilt > 55.0 and inner_lo - 0.004 < wx < inner_hi + 0.004
                        and abs(wy - cy) < 0.012 and not self.flew)
            # SMOOTH-RAMP reward (training-technique doc §3): violence scores
            # BELOW failure — high forces can kick the part out of the area,
            # ending the episode chain; a gentle miss allows a retry.
            escape = bool(self.flew or abs(wx - cx) > 0.05
                          or clearance < -0.03)
            smooth = 0.3 * self.max_tilt / 90.0 + 0.7 * self.lift_tilt / 90.0
            violence = 3.0 * min(max((self.max_dtilt - 20.0) / 30.0, 0.0), 1.0)
            if escape:
                r = -1.0
            else:
                r = smooth - violence
                if captured and clearance > 0.0:
                    r += 1.0 + 1.5 * min(clearance / 0.008, 1.0)
            succ = bool(captured and clearance > 0.003)
            carry_mm = 1000.0 * (wb[4] - SURF_Z) if wb else -99.0
            carry_ok = bool(captured and wb and (wb[4] - SURF_Z) > 0.035)
            if not escape and captured:
                r += 1.0 * min(max(carry_mm / 45.0, 0.0), 1.0)
            return dict(reward=float(r), carry_mm=float(carry_mm),
                        carry_ok=carry_ok,
                        max_tilt=float(self.max_tilt),
                        lift_tilt=float(self.lift_tilt),
                        max_dtilt=float(self.max_dtilt),
                        snap=bool(self.max_dtilt > 25.0),
                        violence=float(violence), escape=escape,
                        tilt=float(tilt),
                        lifted_mm=float(clearance * 1000.0),
                        triggered=bool(self._triggered),
                        captured=bool(captured), flew=bool(self.flew),
                        success=succ,
                        clean=bool(succ and self.max_dtilt <= 25.0))
        lifted = wz - SURF_Z
        captured = (tilt > 55.0 and inner_lo - 0.004 < wx < inner_hi + 0.004
                    and abs(wy - cy) < 0.012 and lifted > 0.008
                    and not self.flew)
        r = 0.3 * self.max_tilt / 90.0 + 0.7 * self.lift_tilt / 90.0
        if captured:
            r += 1.0 + 1.5 * min(max(lifted / 0.02, 0.0), 1.0)
        return dict(reward=float(r), max_tilt=float(self.max_tilt),
                    lift_tilt=float(self.lift_tilt),
                    max_dtilt=float(self.max_dtilt),
                    snap=bool(self.max_dtilt > 25.0),
                    tilt=float(tilt), lifted_mm=float(lifted * 1000.0),
                    captured=bool(captured), flew=bool(self.flew),
                    success=bool(captured and lifted > 0.015))


def cem_update(mu, sig, thetas, rewards, lo, hi, elite_frac=0.34):
    k = max(2, int(round(len(thetas) * elite_frac)))
    idx = np.argsort(rewards)[::-1][:k]
    el = np.asarray(thetas)[idx]
    mu = el.mean(axis=0)
    sig = np.maximum(el.std(axis=0), 0.08 * (hi - lo))
    return mu, sig


def main():
    os.makedirs(OUT, exist_ok=True)
    mats = DEFAULT_MATS
    mp = os.environ.get("GS_PL_MATS", "")
    if mp and os.path.isfile(mp):
        mats = json.load(open(mp))
    parallel = all(m.get("mode") == "parallel" for m in mats)
    LO, HI, MU0 = ((THP_LO, THP_HI, THP_MU0) if parallel
                   else (TH_LO, TH_HI, TH_MU0))
    t_total = (T_TOTAL_P + T_CARRY) if parallel else T_TOTAL

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    import omni.usd
    from isaacsim.core.api import World
    from pxr import UsdGeom, UsdPhysics, Gf

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    # safety ground far below (rigs never reach it)
    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.GetSizeAttr().Set(1.0)
    UsdGeom.XformCommonAPI(g.GetPrim()).SetTranslate(Gf.Vec3d(0, 0, SURF_Z - 0.3))
    UsdGeom.XformCommonAPI(g.GetPrim()).SetScale(Gf.Vec3f(6.0, 2.0, 0.02))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())

    world = World(physics_dt=PHYS_DT, rendering_dt=1.0 / 60.0,
                  stage_units_in_meters=1.0)
    if os.environ.get("GS_PL_GPU", "0") not in ("0", "", "false"):
        try:
            pc = world.get_physics_context()
            pc.enable_gpu_dynamics(True)
            pc.set_broadphase_type("GPU")
            print("[lab] PhysX GPU dynamics ON", flush=True)
        except Exception as e:
            print(f"[lab] GPU dynamics unavailable: {e}", flush=True)
    CTRL_DEC = max(1, int(os.environ.get("GS_PL_CTRL_DEC", "1")))
    if CTRL_DEC > 1:
        print(f"[lab] control decimation x{CTRL_DEC} "
              f"({240 // CTRL_DEC} Hz drive updates)", flush=True)

    log = open(os.path.join(OUT, "evals.jsonl"), "w")
    results = {}

    for mi, mat in enumerate(mats):
        mname = mat["name"]
        print(f"\n[lab] ===== material {mname}: {mat} =====", flush=True)
        # (re)build all rigs for this material
        for r in range(RIGS):
            p = f"/World/Rig{r}"
            if stage.GetPrimAtPath(p).IsValid():
                stage.RemovePrim(p)
        # EE-roll axis: spread each rig's fixed lattice rotation over ±JIT_ROLL so
        # the CEM population evaluates θ across approach angles (rig0 = 0° nominal).
        rig_rolls = ([0.0] if RIGS == 1
                     else [round(float(v), 3) for v in
                           np.linspace(-JIT_ROLL, JIT_ROLL, RIGS)])
        print(f"[lab] rig rolls (deg): {rig_rolls}", flush=True)
        rigs = [PickRig(stage, r, (r * SPACING, 0.0), mat,
                        roll_deg=rig_rolls[r]).build()
                for r in range(RIGS)]
        world.reset()

        all_evals = []
        mu = MU0.copy()
        seed = os.environ.get("GS_PL_SEED_THETA", "")
        if seed:  # warm start (e.g. a previous material's best θ)
            mu = np.clip(np.array([float(v) for v in seed.split(",")]),
                         LO, HI)
        sig = 0.25 * (HI - LO)
        best = dict(reward=-1.0)
        hist = []
        steps_total = int(t_total / PHYS_DT)
        for rnd in range(ROUNDS):
            tw0 = time.time()
            ths = []
            for r in range(RIGS):
                th = (mu if (rnd == 0 and r == 0)
                      else np.clip(np.random.normal(mu, sig), LO, HI))
                ths.append(np.asarray(th))
                jit = tuple(np.random.uniform(-h, h) for h in SEARCH_JIT)                     if parallel else (0.0, 0.0, 0.0)
                rigs[r].reset(th, jitter=jit)
            # settle + run the maneuver, all rigs simultaneously
            dbg = os.environ.get("GS_PL_DEBUG", "0") != "0"
            for s in range(steps_total):
                t = s * PHYS_DT
                if s % CTRL_DEC == 0:
                    for rig in rigs:
                        rig.drive(t)
                world.step(render=False)
                if t > T_SET and s % 12 == 0:
                    for rig in rigs:
                        rig.observe(t)
                if dbg and s % 48 == 0:
                    from pxr import UsdGeom as _UG
                    (wx, wy, wz), tl = rigs[0].washer_state()
                    bb = _UG.BBoxCache(0, [_UG.Tokens.default_]).ComputeWorldBound(
                        stage.GetPrimAtPath(rigs[0].wpath)).ComputeAlignedBox()
                    bbz = (float(bb.GetMax()[2]) - float(bb.GetMin()[2])) * 1000
                    xc = _UG.XformCache()
                    az = float(xc.GetLocalToWorldTransform(
                        rigs[0].primA).ExtractTranslation()[2])
                    bp = xc.GetLocalToWorldTransform(
                        stage.GetPrimAtPath(rigs[0].base + "/FingerB")
                    ).ExtractTranslation()
                    print(f"[dbg] t={t:4.2f} tilt={tl:5.1f} bbz={bbz:4.1f} "
                          f"wz={1000*(wz-SURF_Z):+5.1f} "
                          f"wx={1000*(wx-rigs[0].c[0]):+5.1f} "
                          f"Az={1000*(az-FH/2-SURF_Z):+5.1f} "
                          f"Bx={1000*(float(bp[0])-rigs[0].c[0]):+5.1f} "
                          f"Bz={1000*(float(bp[2])-FH/2-SURF_Z):+5.1f}", flush=True)
            scores = [rig.final_score() for rig in rigs]
            rews = [s_["reward"] for s_ in scores]
            for r, (th, sc) in enumerate(zip(ths, scores)):
                rec = dict(material=mname, round=rnd, rig=r,
                           theta=[round(float(v), 5) for v in th], **sc)
                log.write(json.dumps(rec) + "\n")
                all_evals.append((np.asarray(th), sc["reward"]))
                if sc["reward"] > best["reward"]:
                    best = dict(theta=[float(v) for v in th], **sc)
            log.flush()
            mu, sig = cem_update(mu, sig, ths, rews, LO, HI)
            hist.append(dict(round=rnd, mean=float(np.mean(rews)),
                             max=float(np.max(rews)),
                             succ=sum(1 for s_ in scores if s_["success"]),
                             carry=sum(1 for s_ in scores
                                       if s_.get("carry_ok")),
                             snaps=sum(1 for s_ in scores if s_.get("snap")),
                             theta=[float(v) for v in
                                    ths[int(np.argmax(rews))]]))
            print(f"[lab] {mname} rnd{rnd}: mean={np.mean(rews):.2f} "
                  f"max={np.max(rews):.2f} succ={hist[-1]['succ']}/{RIGS} "
                  f"snaps={hist[-1]['snaps']} "
                  f"tilt={max(s_['max_tilt'] for s_ in scores):.0f} "
                  f"({time.time()-tw0:.0f}s)", flush=True)
            # Echo best-so-far θ every round so a crash still leaves the winner in
            # the logs (evals.jsonl is only recoverable if GS_PL_OUT persists).
            if "theta" in best:
                print(f"[lab] {mname} rnd{rnd} BESTθ r={best['reward']:.2f} "
                      f"succ={best.get('success')} snap={best.get('snap')} "
                      f"theta=[{','.join(f'{v:.5f}' for v in best['theta'])}]",
                      flush=True)
        results[mname] = dict(material=mat, best=best, history=hist)
        print(f"[lab] {mname} BEST reward={best['reward']:.2f} "
              f"captured={best.get('captured')} lifted={best.get('lifted_mm', 0):.1f}mm "
              f"theta={[round(v, 4) for v in best.get('theta', [])]}", flush=True)

        # ── jitter GAUNTLET: robust selection over the top candidates ────────
        # Same theta on ALL rigs, each rig a different jitter from the fixed
        # panel; score = worst-2-mean. Selection stops rewarding lucky runs.
        if parallel and GAUNT_K > 0:
            gtf = os.environ.get("GS_PL_GAUNT_THETAS", "")
            if gtf and os.path.isfile(gtf):
                # explicit candidate list (e.g. re-run for eye-diagram traces)
                cands = [np.asarray(v, dtype=np.float64)
                         for v in json.load(open(gtf))]
            else:
                cands, seen_th = [], []
                for rec in sorted(all_evals, key=lambda e: -e[1]):
                    th = rec[0]
                    span = HI - LO
                    if any(np.linalg.norm((th - o) / span) < 0.15
                           for o in seen_th):
                        continue
                    seen_th.append(th)
                    cands.append(th)
                    if len(cands) >= GAUNT_K:
                        break
            gaunt = []
            tracing = os.environ.get("GS_PL_TRACE", "0") != "0"
            for ci, th in enumerate(cands):
                for r in range(RIGS):
                    rigs[r].reset(th, jitter=PANEL[r % len(PANEL)])
                traces = [[] for _ in range(RIGS)]
                for s_ in range(steps_total):
                    t = s_ * PHYS_DT
                    if s_ % CTRL_DEC == 0:
                        for rig in rigs:
                            rig.drive(t)
                    world.step(render=False)
                    if t > T_SET and s_ % 12 == 0:
                        for r, rig in enumerate(rigs):
                            tl = rig.observe(t)
                            if tracing:
                                if os.environ.get("GS_PL_TRACE") == "2":
                                    (_wx, _wy, _wz), _ = rig.washer_state()
                                    traces[r].append(
                                        (round(t, 3), round(rig._vc, 4),
                                         round(float(tl), 2),
                                         round(1000 * (_wz - SURF_Z), 2)))
                                else:
                                    traces[r].append(
                                        (round(t, 3), round(rig._vc, 4),
                                         round(float(tl), 2)))
                scores = [rig.final_score() for rig in rigs]
                rews = sorted(s_["reward"] for s_ in scores)
                robust = float(np.mean(rews[:2]))          # worst-2-mean
                succ = sum(1 for s_ in scores if s_["success"])
                g = dict(cand=ci, theta=[float(v) for v in th],
                         robust=robust, mean=float(np.mean(rews)),
                         min=rews[0], succ=succ, n=RIGS,
                         panel=[dict(jit=list(PANEL[r % len(PANEL)]),
                                     **{k: scores[r][k] for k in
                                        ("reward", "success", "lifted_mm",
                                         "tilt", "max_dtilt")},
                                     **({"trace": traces[r]} if tracing else {}))
                                for r in range(RIGS)])
                gaunt.append(g)
                log.write(json.dumps(dict(material=mname, gauntlet=ci, **{
                    k: g[k] for k in ("theta", "robust", "mean", "min", "succ")}
                )) + "\n")
                log.flush()
                print(f"[gaunt] {mname} cand{ci}: robust={robust:.2f} "
                      f"mean={g['mean']:.2f} succ={succ}/{RIGS} "
                      f"th={[round(v, 4) for v in th]}", flush=True)
            gaunt.sort(key=lambda g: -g["robust"])
            results[mname]["gauntlet"] = gaunt
            if gaunt:
                w = gaunt[0]
                print(f"[gaunt] {mname} ROBUST WINNER cand{w['cand']} "
                      f"robust={w['robust']:.2f} succ={w['succ']}/{RIGS} "
                      f"theta={[round(v, 4) for v in w['theta']]}", flush=True)

    log.close()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for mname, res in results.items():
            xs = [h["round"] for h in res["history"]]
            ax.plot(xs, [h["max"] for h in res["history"]], "-o", ms=4,
                    label=f"{mname} max")
            ax.plot(xs, [h["mean"] for h in res["history"]], "--", alpha=0.5,
                    label=f"{mname} mean")
        ax.set_xlabel("CEM round")
        ax.set_ylabel("reward")
        ax.set_title("pick_lab: press-drag-rollup learning per material")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "learning_curves.png"), dpi=130)
    except Exception as e:
        print(f"[lab] plot skipped: {e}", flush=True)

    print(f"[lab] DONE -> {OUT}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
