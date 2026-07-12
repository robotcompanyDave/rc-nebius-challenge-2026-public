"""
Slim grasp + sort controller — headless, gateway-free.

A faithful port of the proven grasp/sort logic from the private platform
(`targets/ur10e/adapter.py`), stripped of everything tied to the live gateway
(entity ids, UI selects, telemetry queues, the BaseAdapter base class, cameras,
TCP). What's kept is the IP that matters:

  • the tuned constants (heights, forces, close speed, stall detection, tolerances)
  • the force-feedback close (`grasp_close_fb`) — close until the commanded-vs-actual
    finger LAG or a true stall says the part is between the jaws, then HOLD
  • the pick→place sequencer and the nut/bolt sort planner
  • the world↔base goal conversion + hand-rolled DLS IK drive (via kinematics.py)

Simplification vs the Robotiq port: a PARALLEL-JAW gripper closes in a straight
line, so the 2F-85 fingertip *arc* compensation collapses to holding a fixed
grip-Z while the jaws close — no `_grasp_track_fingertip`.

Provenance line refs are to adapter.py @ feat/lite6-physics-grasp.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Set GS_DEBUG=1 to print per-grasp geometry diagnostics (jaw/wrist/part poses,
# FK-TCP vs goal) — the instrumentation used to bring up Step 0 on the GPU box.
_GS_DEBUG = os.environ.get("GS_DEBUG", "0") not in ("0", "", "false", "False")

from .kinematics import (
    forward_kinematics_full, solve_ik_pose_step,
    JOINT_LOWER, JOINT_UPPER, NUM_JOINTS, HOME_POSITION,
)
from .gripper_robotiq import arc_drop_at_width, fingertip_drop

# ── tuned constants (ported from adapter.py:115–248) ───────────────────────
APPROACH_DH = 0.12          # travel/approach tool-tip height above the platform (m)
GRASP_DH = 0.002            # grasp tool-tip height above the surface (m)
PLACE_DH = 0.020            # release height (m)
GRASP_TIP_FLOOR = 0.001     # min tool-tip height above platform (m)
GRASP_SEAT = -0.004         # tool-Y inset where the part seats in the jaws (m)

FMAX = 40.0                 # finger drive max force (N)
FTARGET = 8.0               # gentle hold clamp (N)
CLOSE_SPEED = 0.6           # openness/s while closing (slow, like a careful pinch)
STALL = 0.03                # commanded-vs-actual openness LAG that signals contact
STALL_MIN = 0.40            # ignore stall below this commanded openness
STALL_DV = 0.004            # per-tick actual advance counted as "stopped"
STUCK_N = 3                 # consecutive stopped ticks = contact
GRASP_SETTLE = 45           # dwell ticks at the descend pose before closing (arm catch-up)

EE_SPEED = 0.06             # m/s autonomous move (delicate phases)
LIFT_SPEED = 0.10           # m/s lift
FAST_MULT = 5.0             # speed × for non-delicate moves
ANG_SPEED = float(np.radians(60.0))   # rad/s autonomous rotate
GRASP_TRACK_SPEED = 0.5     # m/s during close
DECEL_ZONE = 0.04           # m — taper descend speed within this of the goal
DECEL_MIN = 0.01            # m/s floor while decelerating into the surface

ARRIVE_POS_TOL = 0.012      # m — EE-vs-goal arrival
ARRIVE_ROT_TOL = 0.14       # rad (~8°) on the approach axis
DESCEND_POS_TOL = 0.005     # tighter arrival for the descend
DESCEND_ROT_TOL = float(np.radians(30.0))

MAX_JOINT_SPEED = 3.0       # rad/s cap (smoothness)
MAX_RETRIES = 2
SORT_MAX_ATTEMPTS = 3

# zones in BASE frame, split in BASE-Y around the work anchor (adapter.py:230).
# Three lanes now: nuts at −DY, bolts at +DY, washers beyond the bolts at +2·DY.
ZONE_DY = 0.18
ZONE_HALF = 0.09
GRID_COLS = 2
GRID_LAT = 0.04
GRID_FWD = 0.045

# ── pick strategies (mirror of the platform adapter's tilt / press-for-lip) ──
PICK_TILT_DEG = float(os.environ.get("GS_PICK_TILT", "8.0"))
LIP_EDGE_FRAC = 0.6          # press point along +u, fraction of part half-width
LIP_RAISE_DH = 0.04          # hop height between press and re-approach (m)
DRAG_Z = 0.016               # closed-tip plow height for the drag (tool dh, m)
DRAG_DIST = 0.06             # default drag length (m)
# tip c-c opening ≈ part grip width + both pad half-thicknesses (measured: an
# 18 mm nut gap ≈ 34 mm tip c-c → +16 mm) — feeds the closing-arc drop model
PAD_CC_OFFSET = 0.016
ARC_CLEARANCE = 0.0015       # predictive z-comp clearance above the grip point

# ── press-lift washer pick (probe_press_lift.py stage 1, 2026-07-02) ─────────
# ONE finger presses the near rim into the soft pad and HOLDS: the pad pivots
# the washer (sustained 10–12 mm far-rim lip at spread 1.0–1.1, press point at
# 0.9–1.0 of the rim radius). Then the jaw closes WHILE the TCP shifts toward
# the press point in equal proportion, so the pressing finger stays pinned and
# the closing finger sweeps under the raised far rim.
PW_OPEN = 0.35               # jaw openness during the press (0=open, 1=closed)
PW_EDGE_FRAC = 0.95          # press radius fraction (stable pocket 0.9–1.0)
PW_HOLD = 15                 # ticks at press depth before the close STARTS —
                             # short on purpose: the flip-and-lean is
                             # stochastic (A/B 0038/0040-press_lift: identical
                             # trials pop or sustain), so the closing finger
                             # must already be sweeping in when the washer
                             # flips, to catch it either way
PW_CLOSE_SPEED = 0.5         # openness/s during the coordinated close (the
                             # sweep must cover ~30 mm within the pop window)
PW_HOVER_DH = 0.030          # hover height for the position/servo phase
PW_CLEAR = 0.008             # capture margin under the far rim (m)
PW_TILT_DEG = 0.0            # jaw tilt (press finger low). 0 = LEVEL: the flat
                             # pad face is what forms the lip (stage 1); a
                             # tilted pad presses a corner that slips off the
                             # rim — runs 10–13 formed no lip. The closing
                             # finger's low sweep is survivable: run-9 data
                             # shows it PLOWS the washer's far rim upward as it
                             # arrives (bow wave), it does not trench it.
PW_PRESS_INSET = 0.006       # servo the press finger BBOX this far inside the
                             # rim point: the tilted pad's low CONTACT corner
                             # sits ~half a pad-thickness outboard of the bbox
                             # centre (run 10 pressed bare pad outside the rim)
PW_START_CLR = 0.055         # pre-close gap margin over the washer OD: keeps the
                             # closing finger's own dent well clear of the far-rim
                             # shoulder during the press (dent radius ~7 mm; a
                             # 16 mm clearance trenched the washer flat)
# Press-lift MODE (GS_PW_MODE): the stage-1 curves (0038/0039-press_lift) show
# the current soft model produces NO static lip (far rim caps ~1.5 mm from a
# 5 mm dent) — instead the pressed rim SNAPS out at a discrete instant and the
# washer flips, sometimes landing standing/leaning (a 40–60% grasp) instead of
# flat (2.4%). "reorient" exploits that: press → wait for the snap → retreat →
# re-observe → NORMAL grasp of the reoriented washer. "sweep" is the
# coordinated pinned close (David's technique) — it needs a FINGERNAIL tip to
# form a true quasi-static lip; kept for that follow-up.
PW_MODE = os.environ.get("GS_PW_MODE", "reorient")
PW_SNAP_WAIT = 150           # max press-hold ticks waiting for the snap (run
                             # 20: snaps fire within ~ticks of full depth or
                             # not at all — don't wait 5 s)
PW_SNAP_MOVE = 0.006         # part-centre displacement that counts as snapped (m)
PW_SNAP_RIM = 3.5            # far-rim rise that counts as snapped (mm)

# tool +Y (approach) points straight DOWN; columns are tool [X, Y(approach), Z]
_GRASP_R_WORLD = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, -1.0, 0.0]])
_APPROACH_COL = 1           # tool approach axis = FK column 1 (EE-local +Y)

_SEQ_TIMEOUT = {
    "approach_pick": 35.0, "descend_pick": 20.0, "grasp": 14.0, "lift": 20.0,
    "traverse": 40.0, "descend_place": 20.0, "release": 2.5, "retreat": 20.0,
    # lip-press prelude (soft surface): press far edge → hop → re-approach
    "lp_approach": 25.0, "lp_press": 12.0, "lp_raise": 8.0,
    # press-lift: servo a finger onto the rim → press+hold → coordinated close
    "pw_position": 30.0, "pw_press": 25.0, "pw_close": 18.0,
    "pw_snap": 12.0, "pw_regrip": 20.0,
    # drag-separate prelude: closed-jaw plow through the clump
    "dc_approach": 25.0, "dc_descend": 12.0, "dc_drag": 15.0, "dc_lift": 8.0,
}


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _matrix_to_quat(R: np.ndarray):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return w, x, y, z


@dataclass
class GraspOutcome:
    success: bool = False
    lifted_mm: float = 0.0
    grasp_force_N: float = 0.0
    clamp_openness: float = 0.0
    slip_mm: float = 0.0
    fail_reason: Optional[str] = None


@dataclass
class SortResult:
    n_parts: int = 0
    n_correct: int = 0
    attempts: int = 0
    per_part: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return (self.n_correct / self.n_parts) if self.n_parts else 0.0


class GraspSortController:
    def __init__(self, env):
        self.env = env
        self.arm = env.arm
        self.table_top = env.table_top_z
        self.ctrl = self.arm.controller         # kinematics UR10Controller
        self.dt = 1.0 / 60.0
        # Robotiq 2F-85 vs straight-line parametric jaw: the 2F-85's fingertips
        # ARC ~13.7 mm downward as they close (gripper_robotiq._ARC), so descend
        # heights get the predictive z-comp; the parametric jaw needs none.
        self.robotiq = (self.arm.gconf.get("kind") == "robotiq")
        self.soft = getattr(env, "soft_rig", None)

        self._goal_pos = None                   # base frame
        self._goal_R = None
        self._state = "idle"
        self._t0 = 0.0
        self._ctx: dict = {}
        self._auto = False
        self._grasped = None
        self._grasp_only = False
        self._outcome = GraspOutcome()
        # force-feedback trackers
        self._fb_ticks = 0
        self._stuck = 0
        self._actual_prev = None
        self._clamp_logged = False

        # work anchor = FK tool tip at HOME, in base frame (adapter.py:_work_centre)
        T = forward_kinematics_full(HOME_POSITION, tool_offset=self.ctrl.tool_offset)
        self._base_fwd = float(T[0, 3])
        self._base_y0 = float(T[1, 3])

    # ════════════════════════════════════════════════════════════════════════
    # geometry helpers
    # ════════════════════════════════════════════════════════════════════════
    def _stage(self):
        import omni.usd
        return omni.usd.get_context().get_stage()

    def part_world_pose(self, path):
        from pxr import Usd, UsdGeom
        stage = self._stage()
        prim = stage.GetPrimAtPath(path) if stage else None
        if not prim or not prim.IsValid():
            return None
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return np.array([[m[i][j] for j in range(4)] for i in range(4)]).T

    def part_grasp_z(self, path):
        """World Z of the part geometry CENTRE (world bbox)."""
        from pxr import UsdGeom
        prim = self._stage().GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        wb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                               ).ComputeWorldBound(prim).ComputeAlignedBox()
        return float((wb.GetMin()[2] + wb.GetMax()[2]) / 2.0)

    def _part_long_axis_local(self, path):
        from pxr import UsdGeom
        try:
            prim = self._stage().GetPrimAtPath(path)
            lb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                                   ).ComputeUntransformedBound(prim).ComputeAlignedBox()
            mn, mx = lb.GetMin(), lb.GetMax()
            dims = [float(mx[i] - mn[i]) for i in range(3)]
            e = np.zeros(3); e[int(np.argmax(dims))] = 1.0
            return e
        except Exception:
            return np.array([0.0, 0.0, 1.0])

    def grasp_R(self, path, kind):
        """Straight-down grasp, jaws yaw-aligned: nut → snap to a hex FLAT; bolt →
        ACROSS the shaft; washer → round, yaw free (0; candidates explore it).

        Gripper-aware bolt yaw: on the REAL 2F-85 the jaw-opening axis is the
        tool's Z (world −Y at yaw 0), so 'across the shaft heading φ' is yaw=φ —
        the +90° that the parametric jaw needs lands the 2F-85 jaws PARALLEL to
        the shaft (the end-to-end grip that can't hold; verified live on the
        platform, adapter commit 6707d1d)."""
        pose = self.part_world_pose(path)
        if pose is None:
            return _GRASP_R_WORLD
        Rp = pose[:3, :3]
        if kind == "bolt":
            w = Rp @ self._part_long_axis_local(path)
            if float(w[0] ** 2 + w[1] ** 2) < 1e-6:
                return _GRASP_R_WORLD
            yaw = float(np.arctan2(float(w[1]), float(w[0])))
            if not self.robotiq:
                yaw += np.pi / 2.0
        elif kind == "washer":
            return _GRASP_R_WORLD
        else:
            nut_yaw = float(np.arctan2(Rp[1, 0], Rp[0, 0]))
            period = np.pi / 3.0
            yaw = nut_yaw - period * round(nut_yaw / period)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ _GRASP_R_WORLD

    @staticmethod
    def tilt_R(R0, lead_xy, tilt_deg):
        """Lean a straight-down grasp so the approach tips toward `lead_xy` — the
        fingertip on that side leads a few degrees LOWER and wraps under the part's
        near edge first (platform adapter `_tilt_R`)."""
        u = np.array([float(lead_xy[0]), float(lead_xy[1]), 0.0])
        n = float(np.linalg.norm(u))
        if n < 1e-9 or abs(float(tilt_deg)) < 1e-3:
            return R0
        u /= n
        axis = np.array([u[1], -u[0], 0.0])          # u × ẑ
        return _rodrigues(axis, float(np.radians(tilt_deg))) @ R0

    def nearest_free_dir(self, part):
        """Unit XY direction from `part` pointing AWAY from its nearest neighbour —
        the side with the most room for a leading fingertip / rising lip."""
        pose = self.part_world_pose(part)
        if pose is None:
            return (1.0, 0.0)
        cx, cy = float(pose[0, 3]), float(pose[1, 3])
        best, bd2 = None, float("inf")
        for p in self.env.part_paths:
            if p == part or p == self._grasped:
                continue
            pp = self.part_world_pose(p)
            if pp is None:
                continue
            dx, dy = float(pp[0, 3]) - cx, float(pp[1, 3]) - cy
            d2 = dx * dx + dy * dy
            if d2 < bd2:
                bd2, best = d2, (dx, dy)
        if best is None or bd2 < 1e-8:
            return (1.0, 0.0)
        n = float(np.sqrt(bd2))
        return (-best[0] / n, -best[1] / n)

    def _arm_root(self):
        return self.arm.arm_root_world_matrix()

    def _base_xy_to_world(self, bx, by):
        M = self._arm_root()
        if M is None:
            return (0.0, 0.0)
        zw = M @ np.array([bx, by, 0.0, 1.0])
        return (float(zw[0]), float(zw[1]))

    def work_centre_xy(self):
        return self._base_xy_to_world(self._base_fwd, self._base_y0)

    @staticmethod
    def zone_for_kind(kind: str) -> str:
        return {"nut": "nuts", "bolt": "bolts", "washer": "washers"}.get(kind, "bolts")

    def zone_slot_world_xy(self, zone, slot):
        if zone == "nuts":
            zone_y = self._base_y0 - ZONE_DY
        elif zone == "washers":
            zone_y = self._base_y0 + 2.0 * ZONE_DY
        else:
            zone_y = self._base_y0 + ZONE_DY
        col, row = slot % GRID_COLS, slot // GRID_COLS
        lat = (col - (GRID_COLS - 1) / 2.0) * GRID_LAT
        fwd = self._base_fwd - row * GRID_FWD
        return self._base_xy_to_world(fwd, zone_y + lat)

    def part_zone(self, x, y) -> str:
        M = self._arm_root()
        if M is None:
            return "central"
        try:
            pb = np.linalg.inv(M) @ np.array([x, y, self.table_top, 1.0])
        except np.linalg.LinAlgError:
            return "central"
        by = float(pb[1]) - self._base_y0
        if by < -ZONE_HALF:
            return "nuts"
        if by > ZONE_DY + ZONE_HALF:
            return "washers"
        if by > ZONE_HALF:
            return "bolts"
        return "central"

    # ── goal pose (world tool pose → base frame) ──────────────────────────────
    def _set_goal_world(self, world_pos, world_R) -> bool:
        M = self._arm_root()
        if M is None:
            return False
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return False
        Tw = np.eye(4); Tw[:3, :3] = world_R; Tw[:3, 3] = world_pos
        Tb = Minv @ Tw
        self._goal_pos = Tb[:3, 3].copy()
        self._goal_R = Tb[:3, :3].copy()
        return True

    def _seq_goal(self, xy, dh: float):
        R = self._ctx.get("grasp_R", _GRASP_R_WORLD)
        self._set_goal_world(np.array([xy[0], xy[1], self.table_top + dh]), R)

    def _goal_deviation(self):
        if self._goal_pos is None:
            return 0.0, 0.0
        T = forward_kinematics_full(self.ctrl.joint_positions, tool_offset=self.ctrl.tool_offset)
        pos_dev = float(np.linalg.norm(T[:3, 3] - self._goal_pos))
        R_err = self._goal_R @ T[:3, :3].T
        ang = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
        return pos_dev, ang

    def _approach_axis_error(self) -> float:
        if self._goal_R is None:
            return 0.0
        za = forward_kinematics_full(self.ctrl.joint_positions,
                                     tool_offset=self.ctrl.tool_offset)[:3, _APPROACH_COL]
        zg = self._goal_R[:, _APPROACH_COL]
        d = float(np.dot(za, zg) / (np.linalg.norm(za) * np.linalg.norm(zg) + 1e-9))
        return float(np.arccos(np.clip(d, -1.0, 1.0)))

    # ════════════════════════════════════════════════════════════════════════
    # IK drive (base frame); ported from adapter.py:_solve_ik / _auto_drive_arm
    # ════════════════════════════════════════════════════════════════════════
    def _solve_ik(self, target_pos, target_R, warm, iters=6):
        q = np.asarray(warm, dtype=float).copy()
        tool = self.ctrl.tool_offset
        for _ in range(iters):
            T = forward_kinematics_full(q, tool_offset=tool)
            pos_err = target_pos - T[:3, 3]
            R_err = target_R @ T[:3, :3].T
            ang = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
            if ang > 1e-8:
                skew = (R_err - R_err.T) / 2.0
                axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
                an = float(np.linalg.norm(axis))
                rot_err = (axis / an) * ang if an > 1e-12 else np.zeros(3)
            else:
                rot_err = np.zeros(3)
            dq = solve_ik_pose_step(q, pos_err, rot_err, damping=self.ctrl.damping,
                                    max_step=0.2, tool_offset=tool)
            q = np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)
        return None if np.any(np.isnan(q)) else q

    def _limit_joint_speed(self, q_old, q_new, dt):
        max_step = MAX_JOINT_SPEED * dt
        dq = np.clip(q_new - q_old, -max_step, max_step)
        return q_old + dq

    def _auto_drive(self, dt):
        if self._goal_pos is None:
            return
        q = self.ctrl.joint_positions
        T = forward_kinematics_full(q, tool_offset=self.ctrl.tool_offset)
        cur_pos, cur_R = T[:3, 3], T[:3, :3]
        if self._state in ("approach_pick", "traverse", "retreat"):
            ee_speed, ang_speed = EE_SPEED * FAST_MULT, ANG_SPEED * FAST_MULT
        elif self._state == "lift":
            ee_speed, ang_speed = LIFT_SPEED, ANG_SPEED
        elif self._state == "grasp":
            ee_speed, ang_speed = GRASP_TRACK_SPEED, ANG_SPEED
        else:
            ee_speed, ang_speed = EE_SPEED, ANG_SPEED
        dp = self._goal_pos - cur_pos
        dist = float(np.linalg.norm(dp))
        if (DECEL_ZONE > 0 and dist < DECEL_ZONE
                and self._state in ("descend_pick", "descend_place", "pw_press")):
            ee_speed = max(DECEL_MIN, ee_speed * (dist / DECEL_ZONE))
        step = ee_speed * dt
        tgt_pos = self._goal_pos if dist <= step else cur_pos + dp * (step / dist)
        R_err = self._goal_R @ cur_R.T
        ang = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
        max_ang = ang_speed * dt
        if ang > max_ang and ang > 1e-8:
            skew = (R_err - R_err.T) / 2.0
            axis = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            an = float(np.linalg.norm(axis))
            tgt_R = _rodrigues(axis / an, max_ang) @ cur_R if an > 1e-8 else self._goal_R
        else:
            tgt_R = self._goal_R
        new_q = self._solve_ik(tgt_pos, tgt_R, q)
        if new_q is not None:
            limited = self._limit_joint_speed(q, np.clip(new_q, JOINT_LOWER, JOINT_UPPER), dt)
            self.ctrl.joint_positions = limited
            self.arm.apply_arm(limited)

    # ════════════════════════════════════════════════════════════════════════
    # grasp (physics: real friction hold)
    # ════════════════════════════════════════════════════════════════════════
    def _attach_part(self, path) -> bool:
        """Physics grasp: the part stays a DYNAMIC body held by the closing jaw +
        friction. Record it; verify_pickup tells us if it held (adapter.py:1801)."""
        pose = self.part_world_pose(path)
        if pose is None:
            return False
        self._grasped = path
        return True

    def verify_pickup(self) -> bool:
        if self._grasped is None:
            return False
        pose = self.part_world_pose(self._grasped)
        if pose is None:
            return False
        lifted = float(pose[2, 3]) - self.table_top
        self._outcome.lifted_mm = lifted * 1000.0
        return lifted > 0.025

    def _debug_grasp_geom(self, label=""):
        """Diagnostic (Step 0): print where the jaws / wrist are vs the part, to
        confirm the fingers actually reach + straddle the part. GS_DEBUG-gated."""
        if not _GS_DEBUG:
            return
        from pxr import Usd, UsdGeom
        try:
            stage = self._stage()
            gp = self.arm.gconf.get("gripper_path")

            def wpos(path):
                pr = stage.GetPrimAtPath(path)
                if not pr or not pr.IsValid():
                    return None
                m = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                return [round(float(m[3][i]), 4) for i in range(3)]

            part = self.part_world_pose(self._grasped)
            ppos = [round(float(part[i, 3]), 4) for i in range(3)] if part is not None else None

            # what the controller THINKS the TCP is (FK), and the active goal — in world
            M = self._arm_root()
            q = self.ctrl.joint_positions
            Tb = forward_kinematics_full(q, tool_offset=self.ctrl.tool_offset)
            def to_world(pb):
                if M is None:
                    return None
                w = M @ np.array([pb[0], pb[1], pb[2], 1.0])
                return [round(float(w[i]), 4) for i in range(3)]
            fk_tcp = to_world(Tb[:3, 3])
            goal = to_world(self._goal_pos) if self._goal_pos is not None else None
            wrist = wpos(self.arm.tool_link)
            print(f"[grasp-geom:{label}] part={ppos} table_z={round(self.table_top,4)} "
                  f"wrist={wrist} finger_L={wpos(gp + '/finger_left')} "
                  f"finger_R={wpos(gp + '/finger_right')} base={wpos(gp + '/base')}", flush=True)
            print(f"[grasp-geom:{label}] fk_tcp_world={fk_tcp} goal_world={goal} "
                  f"actual_open={self.arm.gripper_actual_openness()} "
                  f"eff={self.arm.gripper_effort()}", flush=True)
        except Exception as e:
            print(f"[grasp-geom] err {e}", flush=True)

    def _grasp_close_fb(self, dt) -> bool:
        """Close, STOP on contact via commanded-vs-actual finger LAG or a true stall
        (adapter.py:2084). Returns True when clamped (on part) or fully closed (empty)."""
        self._fb_ticks += 1
        if self._fb_ticks == 1:
            self._debug_grasp_geom("start")
        # Settle: let the PHYSICAL arm catch up to the commanded descend pose before
        # closing. The FK sequencer transitions on COMMANDED arrival, but the high-gain
        # joints still lag a few cm; _auto_drive keeps driving to the descend goal
        # during these dwell ticks while we hold the jaws open.
        if self._fb_ticks <= GRASP_SETTLE:
            self.arm.set_gripper(0.0)
            return False
        actual = self.arm.gripper_actual_openness()
        cmd = self.arm.gripper_openness
        lag = (cmd - actual) if actual is not None else 0.0
        adv = (actual - self._actual_prev) if (actual is not None and self._actual_prev is not None) else 1.0
        self._actual_prev = actual
        if actual is not None and cmd > STALL_MIN and adv < STALL_DV:
            self._stuck += 1
        else:
            self._stuck = 0
        contact = cmd > STALL_MIN and (lag >= STALL or self._stuck >= STUCK_N)
        # track peak force
        eff = self.arm.gripper_effort()
        if eff is not None:
            self._outcome.grasp_force_N = max(self._outcome.grasp_force_N, eff)
        if contact or cmd >= 0.98:
            self._outcome.clamp_openness = cmd
            self._debug_grasp_geom("clamp")
            return True
        self.arm.set_gripper(min(1.0, cmd + CLOSE_SPEED * dt))
        return False

    def _ramp_gripper(self, target, dt) -> bool:
        step = 2.0 * dt
        cur = self.arm.gripper_openness
        cur = min(target, cur + step) if cur < target else max(target, cur - step)
        self.arm.set_gripper(cur)
        return abs(self.arm.gripper_openness - target) < 0.02

    # ════════════════════════════════════════════════════════════════════════
    # sequencer
    # ════════════════════════════════════════════════════════════════════════
    def _arrived(self) -> bool:
        pos_dev, _ = self._goal_deviation()
        pos_tol, rot_tol = ARRIVE_POS_TOL, ARRIVE_ROT_TOL
        if self._state == "descend_pick":
            pos_tol, rot_tol = DESCEND_POS_TOL, max(rot_tol, DESCEND_ROT_TOL)
        return pos_dev <= pos_tol and self._approach_axis_error() <= rot_tol

    def _timed_out(self) -> bool:
        return (time.monotonic() - self._t0) > _SEQ_TIMEOUT.get(self._state, 12.0)

    def _advance(self, state):
        self._state = state
        self._t0 = time.monotonic()
        c = self._ctx
        if state == "approach_pick":
            self.arm.set_gripper(0.0)
            self._seq_goal(c["pick_xy"], c.get("approach_dh", APPROACH_DH))
        elif state == "descend_pick":
            # The pick point is the candidate's (part + xy_offset), not the part
            # origin — otherwise the sampled xy miss the scorer must learn is
            # silently undone here. For the lip strategy the pick_xy is already
            # the lip-side point (set by begin_*).
            xy = c["pick_xy"]
            gz = self.part_grasp_z(c["part"])
            dz = (gz - self.table_top) if gz is not None else GRASP_DH
            dz += c.get("grasp_dz", 0.0)            # DR depth perturbation
            # soft surface: the planner may descend to the IK floor BELOW the
            # surface (fingertips wrap under a flat part; 5 mm hard stop backstops)
            floor = (-self.soft.ik_depth) if self.soft is not None else GRASP_TIP_FLOOR
            dz = max(dz, floor)
            if self.robotiq:
                # predictive z-comp (arc_profile): the 2F-85 fingertip drops
                # arc_drop(W) as the jaws close to the part width — descend the
                # TCP that much HIGHER so the closing arc lands the grip AT the
                # part instead of ramming through the surface.
                w_cc = c.get("grip_w", 0.019) + PAD_CC_OFFSET
                dz += arc_drop_at_width(w_cc) + ARC_CLEARANCE
            self._seq_goal(xy, dz)
        elif state == "grasp":
            self._fb_ticks = 0; self._stuck = 0
            self._actual_prev = None; self._clamp_logged = False
            self._attach_part(c["part"])
        elif state == "lift":
            self._seq_goal(c["pick_xy"], APPROACH_DH)
        elif state == "traverse":
            self._seq_goal(c["place_xy"], APPROACH_DH)
        elif state == "descend_place":
            self._seq_goal(c["place_xy"], PLACE_DH)
        elif state == "release":
            self.arm.set_gripper(0.0)
        elif state == "retreat":
            self._seq_goal(c["place_xy"], APPROACH_DH)
        # ── lip-press prelude (soft surface): press far edge → hop → approach ──
        elif state == "lp_approach":
            self.arm.set_gripper(1.0)           # CLOSED — press with the tip pad
            self._seq_goal(c["press_xy"], APPROACH_DH)
        elif state == "lp_press":
            if self.soft is not None:
                self.soft.pressing = True       # part-press coupling ON
            self._seq_goal(c["press_xy"], c.get("press_dh", 0.010))
        elif state == "lp_raise":
            if self.soft is not None:
                self.soft.pressing = False
            self.arm.set_gripper(0.0)           # re-open for the pick
            self._seq_goal(c["press_xy"], LIP_RAISE_DH)
        # ── press-lift: press one finger, hold, close with a pinned pivot ─────
        elif state == "pw_position":
            self.arm.set_gripper(PW_OPEN)
            c["pw_tcp_xy"] = list(c["press_xy"])
            c["pw_tcp_z"] = self.table_top + PW_HOVER_DH
            self._seq_goal(c["press_xy"], PW_HOVER_DH)
        elif state == "pw_press":
            if self.soft is not None:
                self.soft.pressing = True       # part-press coupling ON
            c["pw_hold"] = 0
        elif state == "pw_close":
            self._fb_ticks = 0; self._stuck = 0
            self._actual_prev = None
            self._attach_part(c["part"])
            # freeze the dent (memory foam) and lift the jaw JUST off the hard
            # stop (2 mm → tips at −3): sweeping at −5 dragged the closing
            # finger along the stop at 40+ N (run 18), but raising to −1
            # dropped the leaning washer — the press finger is its support
            # (run 21: two flips, both lost during the raise).
            self._lip_capture_press()
            if self.soft is not None:
                self.soft.pressing = False
                c["pw_tcp_z"] = self.table_top - self.soft.depth + 0.002
            else:
                c["pw_tcp_z"] = self.table_top - 0.001
        # ── drag-separate prelude: closed-jaw plow through the clump ──────────
        elif state == "dc_approach":
            self.arm.set_gripper(1.0)
            self._seq_goal(c["drag_start"], APPROACH_DH)
        elif state == "dc_descend":
            self._seq_goal(c["drag_start"], DRAG_Z)
        elif state == "dc_drag":
            self._seq_goal(c["drag_end"], DRAG_Z)
        elif state == "dc_lift":
            self.arm.set_gripper(0.0)
            self._seq_goal(c["drag_end"], APPROACH_DH)

    def _finish(self, success, reason=None):
        if not success:
            self.arm.set_gripper(0.0)
        if self.soft is not None:
            self.soft.pressing = False
            self.soft.press_foot = None
        self._auto = False
        self._state = "idle"
        self._goal_pos = None; self._goal_R = None
        self._outcome.success = bool(success)
        self._outcome.fail_reason = reason

    def _lip_capture_press(self):
        """Freeze the dent as a phantom foot (memory foam) so the pad holds its
        shape — and the part its tilt — while the gripper hops and re-approaches
        the raised lip. Cleared in _finish."""
        if self.soft is None:
            return
        px, py = self._ctx.get("press_xy", (0.0, 0.0))
        pen = 0.0
        try:
            for (fx, fy, _hw, fpen) in self.soft._feet():
                if (fx - px) ** 2 + (fy - py) ** 2 <= 0.03 ** 2:
                    pen = max(pen, float(fpen))
        except Exception:
            pen = 0.0
        if pen <= 0.0005:
            pen = 0.6 * self.soft.ik_depth
        self.soft.press_foot = (float(px), float(py), 0.008,
                                min(pen, self.soft.depth))

    def _finger_tips(self):
        """[(x, y, bottom_z), ...] world fingertip footprints from the gripper
        finger prims (the soft rig already tracks them)."""
        if self.soft is None:
            return []
        from pxr import UsdGeom
        bc = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        out = []
        for p in self.soft._finger_prims:
            if not p or not p.IsValid():
                continue
            b = bc.ComputeWorldBound(p).ComputeAlignedBox()
            mn, mx = b.GetMin(), b.GetMax()
            out.append((0.5 * float(mn[0] + mx[0]), 0.5 * float(mn[1] + mx[1]),
                        float(mn[2]),
                        0.5 * float(mx[0] - mn[0]), 0.5 * float(mx[1] - mn[1])))
        return out

    def _pw_far_rim_mm(self, c, u, W):
        """Height (mm above the surface) of the washer rim point opposite the
        press direction u — the lip the closing finger must get under."""
        pose = self.part_world_pose(c["part"])
        if pose is None:
            return None
        d = pose[:3, :3].T @ np.array([-u[0], -u[1], 0.0])
        d[2] = 0.0
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return None
        rim = pose[:3, 3] + pose[:3, :3] @ (d / n * W / 2.0)
        return (float(rim[2]) - self.table_top) * 1000.0

    def _pw_tick(self, dt):
        """press-lift chain: servo the press finger onto the rim (pw_position),
        press to depth and hold while the pad pivots the washer (pw_press),
        then close while shifting the TCP so the press finger stays pinned and
        the closing finger sweeps under the raised far rim (pw_close)."""
        c = self._ctx
        P = c["press_xy"]
        u = c["press_u"]
        surface = self.table_top
        tips = self._finger_tips()
        if len(tips) < 2:
            self._finish(False, "pw_no_fingers")
            return
        along = [(t[0] - P[0]) * u[0] + (t[1] - P[1]) * u[1] for t in tips]
        press_tip = tips[int(np.argmax(along))]
        # servo target for the press finger BBOX CENTRE. Level jaw (tilt 0):
        # centre ON the rim point — the flat face press that formed the lip in
        # stage 1. Tilted jaw: inset by the AABB half-extent along u so the low
        # corner lands on the rim (corner presses proved fragile — runs 10–13).
        if PW_TILT_DEG > 0.0:
            half_u = abs(press_tip[3] * u[0]) + abs(press_tip[4] * u[1])
        else:
            half_u = 0.0
        Ps = (P[0] - u[0] * half_u, P[1] - u[1] * half_u)
        state = self._state
        gxy = c["pw_tcp_xy"]

        W = float(c.get("grip_w", 0.024))

        if "pw_prims_logged" not in c:
            c["pw_prims_logged"] = True
            paths = [p.GetPath().pathString for p in self.soft._finger_prims
                     if p and p.IsValid()]
            print(f"[pw] finger prims: {paths}", flush=True)
        c["pw_all"] = c.get("pw_all", 0) + 1
        if c["pw_all"] % (40 if state == "pw_press" else 120) == 1:
            pos_dev, _ = self._goal_deviation()
            ax = self._approach_axis_error()
            tz = [f"{(t[2] - self.table_top) * 1000:.0f}" for t in tips]
            rim = self._pw_far_rim_mm(c, u, W)
            # live washer displacement from where the press point was computed
            off = "?"
            pose = self.part_world_pose(c["part"])
            if pose is not None and "pw_part_xy0" in c:
                dx = float(pose[0, 3]) - c["pw_part_xy0"][0]
                dy = float(pose[1, 3]) - c["pw_part_xy0"][1]
                off = (f"{np.hypot(dx, dy)*1000:.1f}mm"
                       f"(u={ (dx*u[0]+dy*u[1])*1000:+.1f})")
            print(f"[pw] {state}: tips_z={tz} P_along={[f'{a*1000:.0f}' for a in along]} "
                  f"pos_dev={pos_dev*1000:.0f}mm axis={ax:.2f} "
                  f"cmd={self.arm.gripper_openness:.2f} "
                  f"tcp_z_goal={(c['pw_tcp_z'] - self.table_top) * 1000:.0f}mm "
                  f"far_rim={'?' if rim is None else f'{rim:.1f}'}mm "
                  f"part_off={off}",
                  flush=True)

        if state == "pw_position":
            # In-air pre-close to a MEASURED tip gap just wider than the washer:
            # symmetric commanded closing halves the usable stroke once the press
            # finger jams, so the closing finger must START near the far rim
            # (first run began the close at an 83 mm gap and ran out of stroke).
            close_tip = tips[int(np.argmin(along))]
            gap = float(np.hypot(close_tip[0] - press_tip[0],
                                 close_tip[1] - press_tip[1]))
            if gap > W + PW_START_CLR:
                self.arm.set_gripper(min(1.0, self.arm.gripper_openness
                                         + 0.7 * PW_CLOSE_SPEED * dt))
            # gentle servo (gain 0.4 oscillated ±20 mm around P)
            gxy[0] += float(np.clip(0.15 * (Ps[0] - press_tip[0]), -0.0015, 0.0015))
            gxy[1] += float(np.clip(0.15 * (Ps[1] - press_tip[1]), -0.0015, 0.0015))
            self._set_goal_world(np.array([gxy[0], gxy[1], c["pw_tcp_z"]]),
                                 c["grasp_R"])
            miss = float(np.hypot(Ps[0] - press_tip[0], Ps[1] - press_tip[1]))
            pos_dev, _ = self._goal_deviation()
            axis_ok = self._approach_axis_error() < 0.10   # jaw pointing DOWN —
            # without this gate a mid-rotation jaw (near-horizontal) satisfies
            # the xy-gap check while the tips are still high in the air
            if (miss < 0.0025 and gap <= W + PW_START_CLR + 0.003
                    and pos_dev < 0.008 and axis_ok):
                self._advance("pw_press")
            elif self._timed_out():
                self._finish(False, "timeout:pw_position")
            return

        if state == "pw_press":
            # xy FROZEN — chasing the press point during the descent nudges the
            # washer off the press (run 1 pressed bare pad). Straight down only,
            # with a FIXED goal so _auto_drive's DECEL_ZONE gives a gentle final
            # contact (run 16: a constant-rate z-servo marched through contact
            # at 60 mm/s and squirted the washer 5 mm out from under the pad —
            # stage 1's press worked because descend decel tapered to 10 mm/s).
            depth = self.soft.depth if self.soft is not None else 0.005
            z_target = surface - depth
            err = press_tip[2] - z_target            # >0 → not deep enough yet
            c["pw_tcp_z"] = z_target - 0.003         # constant, just past depth
            self._set_goal_world(np.array([gxy[0], gxy[1], c["pw_tcp_z"]]),
                                 c["grasp_R"])
            # pressed = at depth, OR tip stalled below the surface (a partial
            # press on the part still forms the lip — stage 1)
            tip_prev = c.get("pw_tip_prev")
            c["pw_tip_prev"] = press_tip[2]
            stalled = (tip_prev is not None
                       and abs(press_tip[2] - tip_prev) < 0.00005
                       and press_tip[2] < surface - 0.001)
            if err < 0.0008 or stalled:
                c["pw_hold"] += 1
                if c["pw_hold"] >= PW_HOLD:
                    self._advance("pw_snap" if PW_MODE == "reorient"
                                  else "pw_close")
                    return
            if self._timed_out():
                # a partial press is still a press — proceed
                self._advance("pw_snap" if PW_MODE == "reorient" else "pw_close")
            return

        if state == "pw_snap":
            # keep pressing and WAIT for the stochastic snap-flip (stage-1
            # curves: a single-tick energy release ~0.5 s after full depth).
            # A small z pump agitates the release (run 20: 4/6 presses never
            # snapped on a static hold).
            c["pw_snap_t"] = c.get("pw_snap_t", 0) + 1
            pump = 0.0015 * float(np.sin(c["pw_snap_t"] / 10.0))
            self._set_goal_world(np.array([gxy[0], gxy[1],
                                           c["pw_tcp_z"] + pump]),
                                 c["grasp_R"])
            rim = self._pw_far_rim_mm(c, u, W)
            moved = None
            pose = self.part_world_pose(c["part"])
            if pose is not None and "pw_part_xy0" in c:
                moved = float(np.hypot(pose[0, 3] - c["pw_part_xy0"][0],
                                       pose[1, 3] - c["pw_part_xy0"][1]))
            flipped = rim is not None and rim > PW_SNAP_RIM
            slid = moved is not None and moved > PW_SNAP_MOVE
            if flipped or slid or c["pw_snap_t"] >= PW_SNAP_WAIT or self._timed_out():
                print(f"[pw] snap: {'FLIP' if flipped else ('slide' if slid else 'no')} "
                      f"after {c['pw_snap_t']} ticks (rim={rim}, moved="
                      f"{None if moved is None else round(moved*1000,1)}mm)",
                      flush=True)
                if flipped:
                    # the washer is UP, leaning on the press finger — do NOT
                    # retreat (removing the support drops it flat again, run
                    # 20 #2): sweep the closing finger in and clamp it NOW
                    self._advance("pw_close")
                else:
                    if self.soft is not None:
                        self.soft.pressing = False
                    self._advance("pw_regrip")
            return

        if state == "pw_regrip":
            # retreat above the LIVE part, then re-enter the NORMAL grasp path
            # on whatever pose the snap produced (standing/leaning ≈ 40–60%
            # grasp vs 2.4% flat)
            pose = self.part_world_pose(c["part"])
            if pose is None:
                self._finish(False, "pw_part_lost")
                return
            live_xy = (float(pose[0, 3]), float(pose[1, 3]))
            self.arm.set_gripper(0.0)
            self._set_goal_world(np.array([live_xy[0], live_xy[1],
                                           surface + APPROACH_DH]),
                                 c["grasp_R"])
            pos_dev, _ = self._goal_deviation()
            if pos_dev < 0.015:
                kind = self.env.part_kinds.get(c["part"], "washer")
                c["pick_xy"] = live_xy
                c["grasp_R"] = self.grasp_R(c["part"], kind)
                c["grasp_dz"] = 0.0
                c["strategy"] = "press_reorient"
                self._advance("approach_pick")
            elif self._timed_out():
                self._finish(False, "timeout:pw_regrip")
            return

        if state == "pw_close":
            # The commanded-vs-actual LAG detector is unusable here: the PRESS
            # finger is intentionally jammed, so actual lags from tick one and
            # fires a false "contact" the moment cmd crosses STALL_MIN. Detect
            # capture GEOMETRICALLY instead: the closing tip has swept inside
            # the washer's rim circle (tip gap < OD·0.9), then squeeze briefly
            # for a firm diagonal clamp and lift.
            close_tip = tips[int(np.argmin(along))]
            gap = float(np.hypot(close_tip[0] - press_tip[0],
                                 close_tip[1] - press_tip[1]))
            prev_cmd = self.arm.gripper_openness
            cmd = min(1.0, prev_cmd + PW_CLOSE_SPEED * dt)
            self.arm.set_gripper(cmd)
            # Pin the press finger with a FEEDFORWARD base shift: the commanded
            # symmetric close moves each commanded finger by δ·half_stroke, so
            # shifting the TCP by exactly that keeps the press finger's command
            # world-fixed (no friction fight) and the closing finger sweeps at
            # 2× joint rate — the "correct proportion". (Feedback-only pinning
            # never engaged: the pinned finger has no error, so the base never
            # moved and the close ran out of stroke.)
            half_stroke = float(self.arm.gconf.get("stroke_mm", 85.0)) / 2000.0
            ff = (cmd - prev_cmd) * half_stroke
            gxy[0] += u[0] * ff + 0.25 * (Ps[0] - press_tip[0])
            gxy[1] += u[1] * ff + 0.25 * (Ps[1] - press_tip[1])
            self._set_goal_world(np.array([gxy[0], gxy[1], c["pw_tcp_z"]]),
                                 c["grasp_R"])
            eff = self.arm.gripper_effort()
            if eff is not None:
                self._outcome.grasp_force_N = max(self._outcome.grasp_force_N, eff)
            c["pw_dbg"] = c.get("pw_dbg", 0) + 1
            if c["pw_dbg"] % 15 == 1:
                actual = self.arm.gripper_actual_openness()
                rim_z = self._pw_far_rim_mm(c, u, W)
                print(f"[pw] close: gap={gap*1000:.1f}mm cmd={cmd:.2f} "
                      f"act={actual if actual is None else round(actual, 2)} "
                      f"ptip_z={(press_tip[2]-self.table_top)*1000:.1f} "
                      f"ctip_z={(close_tip[2]-self.table_top)*1000:.1f} "
                      f"far_rim={'?' if rim_z is None else f'{rim_z:.1f}'}mm",
                      flush=True)
            if c.get("pw_captured"):
                c["pw_squeeze"] = c.get("pw_squeeze", 0) + 1
                if c["pw_squeeze"] >= 20:
                    self._outcome.clamp_openness = cmd
                    self._lip_capture_press()    # memory-foam the dent
                    if self.soft is not None:
                        self.soft.pressing = False
                    self._record_slip_ref()
                    c["pick_xy"] = (gxy[0], gxy[1])   # lift straight up
                    self._advance("lift")
                return
            # CAPTURE = the tip gap STALLS inside the physical grip band.
            # Tip-CENTRE gap when gripping ≈ part width + both pad halves
            # (PAD_CC_OFFSET ≈ 16 mm) — the old `gap < W·0.9` threshold was
            # physically unreachable while holding a part, so run 9's stall at
            # 34 mm/172 N (a diagonal clamp) was mislabeled "blocked".
            # band floor 12 mm: a washer clamped VERTICAL (flipped, leaning)
            # reads ~ thickness + both pad halves ≈ 18 mm tip-centre gap
            band_lo = 0.012
            band_hi = W + PAD_CC_OFFSET + 0.006
            last = c.get("pw_close_gap")
            c["pw_close_gap"] = gap
            stalled_gap = last is not None and (last - gap) < 0.0003
            if stalled_gap:
                self._stuck += 1
            else:
                self._stuck = 0
            if self._stuck >= STUCK_N * 3:
                if band_lo <= gap <= band_hi:
                    c["pw_captured"] = True      # clamped on the washer
                else:
                    self._finish(False, "pw_blocked_outside")
                return
            if cmd >= 0.995:
                if band_lo <= gap <= band_hi:
                    c["pw_captured"] = True
                else:
                    self._finish(False, "pw_closed_empty")
                return
            if self._timed_out():
                if band_lo <= gap <= band_hi:
                    c["pw_captured"] = True      # slow clamp still a clamp
                else:
                    self._finish(False, "timeout:pw_close")
            return

    def _tick(self, dt):
        if not self._auto or self._state in ("idle", "done", "error"):
            return
        state = self._state
        # ── press-lift chain (drives its own goal servo + gripper) ────────────
        if state.startswith("pw_"):
            self._pw_tick(dt)
            return
        # ── prelude chains (own gripper handling + custom timeout semantics) ──
        if state.startswith("lp_") or state.startswith("dc_"):
            self._ramp_gripper(1.0 if state in ("lp_approach", "lp_press",
                                                "dc_approach", "dc_descend", "dc_drag")
                               else 0.0, dt)
            if state == "lp_press" and (self._arrived() or self._timed_out()):
                # a partial press stopped by the hard stop is still a usable press
                self._lip_capture_press()
                self._advance("lp_raise")
                return
            if self._timed_out():
                self._finish(False, "timeout:" + state)
                return
            if self._arrived():
                nxt = {"lp_approach": "lp_press", "lp_raise": "approach_pick",
                       "dc_approach": "dc_descend", "dc_descend": "dc_drag",
                       "dc_drag": "dc_lift", "dc_lift": "approach_pick"}[state]
                self._advance(nxt)
            return
        if state == "grasp":
            grip_reached = self._grasp_close_fb(dt)
        else:
            grip_reached = self._ramp_gripper(0.0 if state == "release" else
                                              (1.0 if state == "grasp" else self.arm.gripper_openness), dt)
        if self._timed_out():
            self._finish(False, "timeout:" + state)
            return
        if state == "grasp":
            if self._grasped is None:
                self._finish(False, "grasp_failed"); return
            if grip_reached:
                self._record_slip_ref()
                self._advance("lift")
            return
        if state == "release":
            if grip_reached:
                self._grasped = None
                self._advance("retreat")
            return
        if self._arrived():
            if state == "lift":
                ok = self.verify_pickup()
                self._compute_slip()
                if self._grasp_only:
                    self._finish(ok, None if ok else "dropped_on_lift")
                    return
                if not ok:
                    self._finish(False, "dropped_on_lift")
                    return
                self._advance("traverse")
            elif state == "approach_pick":
                self._advance("descend_pick")
            elif state == "descend_pick":
                self._advance("grasp")
            elif state == "traverse":
                self._advance("descend_place")
            elif state == "descend_place":
                self._advance("release")
            elif state == "retreat":
                self._finish(True, None)

    def _record_slip_ref(self):
        """Part position in tool frame at clamp — slip reference (adapter slip metric)."""
        pose = self.part_world_pose(self._grasped) if self._grasped else None
        self._ctx["_clamp_part_xy"] = (float(pose[0, 3]), float(pose[1, 3])) if pose is not None else None

    def _compute_slip(self):
        ref = self._ctx.get("_clamp_part_xy")
        pose = self.part_world_pose(self._grasped) if self._grasped else None
        if ref and pose is not None:
            self._outcome.slip_mm = float(np.hypot(pose[0, 3] - ref[0], pose[1, 3] - ref[1])) * 1000.0

    # ════════════════════════════════════════════════════════════════════════
    # public API
    # ════════════════════════════════════════════════════════════════════════
    @property
    def state(self) -> str:
        return self._state

    @property
    def busy(self) -> bool:
        return self._auto

    @property
    def outcome(self) -> GraspOutcome:
        return self._outcome

    def force_finish(self, reason="tick_budget"):
        if self._auto:
            self._finish(False, reason)

    def reset_to_home(self):
        self.arm.go_home()
        self._auto = False
        self._state = "idle"
        self._goal_pos = None; self._goal_R = None
        self._grasped = None

    def step(self, dt: Optional[float] = None):
        """Advance the controller one tick. Call BEFORE env.step() so the joint/
        gripper targets are set for the physics step."""
        dt = dt or self.dt
        if self._auto:
            self._tick(dt)
            if self._auto:
                self._auto_drive(dt)

    def _part_grip_width(self, part_path) -> float:
        """Jaw span at grip (m): from the spawn spec (exact dims) or the part's
        horizontal bbox as fallback."""
        try:
            from . import parts as _parts
            spec = self.env.part_specs.get(part_path)
            if spec is not None:
                return _parts.part_grip_width(spec.kind, _parts.size_dims(spec.size))
        except Exception:
            pass
        try:
            from pxr import UsdGeom
            wb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                                   ).ComputeWorldBound(
                self._stage().GetPrimAtPath(part_path)).ComputeAlignedBox()
            mn, mx = wb.GetMin(), wb.GetMax()
            return float(min(mx[0] - mn[0], mx[1] - mn[1]))
        except Exception:
            return 0.019

    def _part_top_dh(self, part_path) -> float:
        """Part's top height above the surface (m) — the lip-press contact height."""
        try:
            from pxr import UsdGeom
            wb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
                                   ).ComputeWorldBound(
                self._stage().GetPrimAtPath(part_path)).ComputeAlignedBox()
            return float(wb.GetMax()[2]) - self.table_top
        except Exception:
            return 0.003

    def _setup_candidate_ctx(self, part_path, kind, candidate: dict) -> str:
        """Fill self._ctx from a candidate action and return the START state.

        Candidate fields (all optional): xy_offset (m,m), grasp_yaw (rad),
        grasp_dz (m), approach_dh (m), strategy ("direct"|"tilt"|"lip"),
        tilt_deg, lead_dir (unit xy), pre_drag ({"dir_xy","dist"} | True).
        The lip strategy needs the soft rig; it degrades to tilt without it."""
        pose = self.part_world_pose(part_path)
        ox, oy = candidate.get("xy_offset", (0.0, 0.0))
        cx, cy = float(pose[0, 3]), float(pose[1, 3])
        pick_xy = (cx + ox, cy + oy)
        if "grasp_yaw" in candidate:
            yaw = candidate["grasp_yaw"]
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ _GRASP_R_WORLD
        else:
            R = self.grasp_R(part_path, kind)
        self._ctx.update({
            "part": part_path, "pick_xy": pick_xy, "grasp_R": R,
            "grasp_dz": candidate.get("grasp_dz", 0.0),
            "approach_dh": candidate.get("approach_dh", APPROACH_DH),
            "grip_w": self._part_grip_width(part_path),
            "strategy": "direct",
        })
        strat = str(candidate.get("strategy", "direct")).lower()
        tilt_deg = float(candidate.get("tilt_deg", PICK_TILT_DEG))
        if strat == "lip" and self.soft is None:
            strat = "tilt"
        if strat == "tilt":
            lead = candidate.get("lead_dir") or self.nearest_free_dir(part_path)
            self._ctx["grasp_R"] = self.tilt_R(self._ctx["grasp_R"], lead, tilt_deg)
            self._ctx["strategy"] = "tilt"
        elif strat == "lip":
            free = candidate.get("lead_dir") or self.nearest_free_dir(part_path)
            u = (-free[0], -free[1])            # press AWAY from free space
            hw_u = 0.5 * self._ctx["grip_w"]
            top_dh = self._part_top_dh(part_path)
            press_d = self.soft.ik_depth
            tip_off = fingertip_drop(1.0) if self.robotiq else 0.0
            self._ctx["press_xy"] = (cx + u[0] * hw_u * LIP_EDGE_FRAC,
                                     cy + u[1] * hw_u * LIP_EDGE_FRAC)
            self._ctx["press_dh"] = max(0.0, top_dh - press_d) + tip_off
            # grip the RAISED lip: jaws close along u, leading fingertip on the
            # lip side (−u), aim slightly lip-ward of centre
            yaw = float(np.arctan2(u[1], u[0])) + (np.pi / 2.0 if self.robotiq else 0.0)
            c2, s2 = float(np.cos(yaw)), float(np.sin(yaw))
            Rz = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
            self._ctx["grasp_R"] = self.tilt_R(Rz @ _GRASP_R_WORLD,
                                               (-u[0], -u[1]), tilt_deg)
            self._ctx["pick_xy"] = (cx - u[0] * hw_u * 0.25,
                                    cy - u[1] * hw_u * 0.25)
            self._ctx["strategy"] = "lip"
        elif strat == "press_lift":
            # press-and-hold pick: needs the soft rig + a flat-ish part; the
            # closing finger sweeps in from the FREE side, so the press goes on
            # the opposite (near) rim. Degrades to tilt when unavailable.
            if self.soft is None:
                lead = candidate.get("lead_dir") or self.nearest_free_dir(part_path)
                self._ctx["grasp_R"] = self.tilt_R(self._ctx["grasp_R"], lead, tilt_deg)
                self._ctx["strategy"] = "tilt"
            else:
                free = candidate.get("lead_dir") or self.nearest_free_dir(part_path)
                u = (-free[0], -free[1])        # press side = away from free space
                hw_u = 0.5 * self._ctx["grip_w"]
                self._ctx["press_u"] = u
                self._ctx["pw_part_xy0"] = (cx, cy)
                self._ctx["press_xy"] = (cx + u[0] * hw_u * PW_EDGE_FRAC,
                                         cy + u[1] * hw_u * PW_EDGE_FRAC)
                # jaw axis along u so one fingertip lands on the press point;
                # tilted so the press finger leads LOW (closing finger clears
                # the pad during the press — level pads trench the washer)
                yaw = float(np.arctan2(u[1], u[0]))
                c2, s2 = float(np.cos(yaw)), float(np.sin(yaw))
                Rz = np.array([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
                # tilt_R lowers the tip OPPOSITE the lead arg (run 9: lead=u
                # put the press finger 17 mm in the air) → lead with −u
                self._ctx["grasp_R"] = self.tilt_R(Rz @ _GRASP_R_WORLD,
                                                   (-u[0], -u[1]), PW_TILT_DEG)
                self._ctx["strategy"] = "press_lift"
        # drag-separate prelude (before any strategy's pick)
        pd = candidate.get("pre_drag")
        if pd:
            if isinstance(pd, dict) and "dir_xy" in pd:
                ux, uy = pd["dir_xy"]
                n = float(np.hypot(ux, uy)) or 1.0
                ux, uy = ux / n, uy / n
                dist = float(pd.get("dist", DRAG_DIST))
            else:
                ux, uy = 1.0, 0.0
                dist = DRAG_DIST
            d = 0.5 * dist
            self._ctx["drag_start"] = (cx - ux * d, cy - uy * d)
            self._ctx["drag_end"] = (cx + ux * d, cy + uy * d)
            return "dc_approach"
        if self._ctx["strategy"] == "press_lift":
            return "pw_position"
        return "lp_approach" if self._ctx["strategy"] == "lip" else "approach_pick"

    def begin_grasp(self, part_path, kind, candidate: Optional[dict] = None):
        """Start a single grasp-only attempt (approach→descend→grasp→lift→verify),
        optionally via a strategy prelude (tilt / lip-press / pre-drag)."""
        candidate = candidate or {}
        pose = self.part_world_pose(part_path)
        if pose is None:
            self._outcome = GraspOutcome(success=False, fail_reason="no_part")
            return
        self._ctx = {}
        start = self._setup_candidate_ctx(part_path, kind, candidate)
        self._grasped = None
        self._outcome = GraspOutcome()
        self._grasp_only = True
        self._auto = True
        self._advance(start)

    def run_until_idle(self, max_ticks: int = 3000) -> GraspOutcome:
        """Drive the env until the current sequence finishes. Steps controller +
        physics each tick."""
        n = 0
        while self._auto and n < max_ticks:
            self.step(self.dt)
            self.env.step(render=False)
            n += 1
        if self._auto:                       # ran out of ticks
            self._finish(False, "tick_budget")
        return self._outcome

    def attempt_grasp(self, part_path, kind, candidate=None, max_ticks=3000) -> GraspOutcome:
        self.begin_grasp(part_path, kind, candidate)
        return self.run_until_idle(max_ticks)

    # ── sort ──────────────────────────────────────────────────────────────────
    def _sort_candidates(self):
        """Ordered [misplaced, unsorted] parts (adapter.py:_sort_plan, 2283)."""
        misplaced, unsorted = [], []
        for p in self.env.part_paths:
            pose = self.part_world_pose(p)
            if pose is None:
                continue
            kind = self.env.part_kinds.get(p, "nut")
            want = self.zone_for_kind(kind)
            zone = self.part_zone(float(pose[0, 3]), float(pose[1, 3]))
            if zone == "central":
                unsorted.append(p)
            elif zone != want:
                misplaced.append(p)
        return misplaced + unsorted

    def begin_pick_place(self, part_path, candidate=None):
        """Full pick→place of one part into its kind's zone, honouring the full
        candidate action (xy miss / yaw / depth / approach / strategy / pre-drag) —
        this is what gives a scorer-guided policy a real lever over the sort. An
        absent candidate falls back to the heuristic defaults."""
        candidate = candidate or {}
        kind = self.env.part_kinds.get(part_path, "nut")
        zone = self.zone_for_kind(kind)
        slot = sum(1 for p in self.env.part_paths if p != part_path
                   and (pp := self.part_world_pose(p)) is not None
                   and self.part_zone(float(pp[0, 3]), float(pp[1, 3])) == zone)
        pose = self.part_world_pose(part_path)
        if pose is None:
            self._outcome = GraspOutcome(success=False, fail_reason="no_part")
            return
        self._ctx = {"zone": zone, "place_xy": self.zone_slot_world_xy(zone, slot)}
        start = self._setup_candidate_ctx(part_path, kind, candidate)
        self._grasped = None
        self._outcome = GraspOutcome()
        self._grasp_only = False
        self._auto = True
        self._advance(start)

    def run_sort_trial(self, policy=None, max_picks=12) -> SortResult:
        """Sort every part into its kind's zone. `policy(part_path, ctrl)->candidate`
        chooses the grasp (None → heuristic grasp_R). Returns the sort metric."""
        res = SortResult(n_parts=len(self.env.part_paths))
        attempts_by_part: dict = {}
        skip: set = set()
        picks = 0
        while picks < max_picks:
            cands = [p for p in self._sort_candidates() if p not in skip]
            if not cands:
                break
            part = cands[0]
            if attempts_by_part.get(part, 0) >= SORT_MAX_ATTEMPTS:
                skip.add(part)              # give up on this part; leave its kind intact
                continue
            attempts_by_part[part] = attempts_by_part.get(part, 0) + 1
            candidate = policy(part, self) if policy else None
            self.begin_pick_place(part, candidate)
            self.run_until_idle()
            self.reset_to_home()
            for _ in range(20):
                self.env.step(render=False)
            picks += 1
        res.attempts = picks
        # final tally
        for p in self.env.part_paths:
            pose = self.part_world_pose(p)
            if pose is None:
                continue
            kind = self.env.part_kinds.get(p, "nut")
            want = self.zone_for_kind(kind)
            zone = self.part_zone(float(pose[0, 3]), float(pose[1, 3]))
            ok = (zone == want)
            res.n_correct += int(ok)
            res.per_part.append({"part": p, "kind": kind, "zone": zone, "correct": ok})
        return res
