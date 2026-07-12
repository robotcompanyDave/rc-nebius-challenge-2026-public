"""
VENDORED — verbatim from the Robot Company `rc-remote-platform` repo
(targets/ur10e/kinematics.py @ branch feat/lite6-physics-grasp). Kept self-contained
(numpy-only) so this public challenge repo carries no dependency on the private platform.
Provenance: UR10/UR10e kinematic constants derive from the official Universal Robots URDF.

Forward Kinematics and Differential IK for Universal Robots UR10 / UR10e.

Implements the kinematic chain from the UR10 URDF:
  base_link → shoulder_pan(Z) → shoulder_link → shoulder_lift(Y) → upper_arm
            → elbow(Y) → forearm → wrist_1(Y) → wrist_1_link
            → wrist_2(Z) → wrist_2_link → wrist_3(Y) → wrist_3_link

Unlike the Lite6/Mecharm (all joints about local Z), the UR has mixed joint
axes (Z, Y, Y, Y, Z, Y).  Each joint therefore carries its own rotation axis,
and FK rotates about that local axis rather than assuming Z.

Transforms (xyz/rpy origins + axes) are taken directly from the URDF
<joint> elements:
    D:/isaacsim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/ur10/urdf/ur10.urdf

The IK solver uses damped least-squares (DLS) — hand-rolled here (NOT Lula) so
the UR matches the Lite6/Mecharm kinematics layer and a shared sort layer can
drive all three arms uniformly.  Both a position Jacobian (3×6, finite
differences) and an analytic angular Jacobian (joint axes in world frame) are
provided so the adapter can do full 6-DOF velocity-mode teleop.
"""

import numpy as np
from typing import Tuple, Optional

# ── Constant transforms from the URDF ──────────────────────────────
# Each entry: (xyz, rpy) from <joint><origin .../> plus the joint <axis>.
# Order matches shoulder_pan … wrist_3.
#
# UR10 kinematic chain (from the official URDF):
#   d1 = 0.1273   (base → shoulder, +Z)
#   a2 = -0.612   (upper arm length, encoded in elbow origin z=0.612)
#   a3 = -0.5723  (forearm length, encoded in wrist_1 origin z=0.5723)
#   d4 = 0.163941 (shoulder→elbow y offsets:  0.220941 - 0.1719 + 0.1149)
#   d5 = 0.1157   (wrist_2 → wrist_3, +Z in its frame)
#   d6 = 0.0922   (wrist_3 → flange/ee_fixed_joint)

_JOINT_PARAMS = [
    # shoulder_pan_joint:  base_link → shoulder_link
    {"xyz": (0.0, 0.0, 0.1273),     "rpy": (0.0, 0.0, 0.0),            "axis": (0.0, 0.0, 1.0)},
    # shoulder_lift_joint: shoulder_link → upper_arm_link
    {"xyz": (0.0, 0.220941, 0.0),   "rpy": (0.0, np.pi / 2, 0.0),      "axis": (0.0, 1.0, 0.0)},
    # elbow_joint:         upper_arm_link → forearm_link
    {"xyz": (0.0, -0.1719, 0.612),  "rpy": (0.0, 0.0, 0.0),            "axis": (0.0, 1.0, 0.0)},
    # wrist_1_joint:       forearm_link → wrist_1_link
    {"xyz": (0.0, 0.0, 0.5723),     "rpy": (0.0, np.pi / 2, 0.0),      "axis": (0.0, 1.0, 0.0)},
    # wrist_2_joint:       wrist_1_link → wrist_2_link
    {"xyz": (0.0, 0.1149, 0.0),     "rpy": (0.0, 0.0, 0.0),            "axis": (0.0, 0.0, 1.0)},
    # wrist_3_joint:       wrist_2_link → wrist_3_link
    {"xyz": (0.0, 0.0, 0.1157),     "rpy": (0.0, 0.0, 0.0),            "axis": (0.0, 1.0, 0.0)},
]

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Joint limits from the URDF <limit lower/upper> — ±2π for all 6.
JOINT_LOWER = np.array([-6.28318, -6.28318, -6.28318, -6.28318, -6.28318, -6.28318])
JOINT_UPPER = np.array([ 6.28318,  6.28318,  6.28318,  6.28318,  6.28318,  6.28318])

NUM_JOINTS = 6

# Home position: a table-ready pose with the gripper pointing STRAIGHT DOWN at
# the work zone (for top-down nut/bolt grasping), NOT the old upright pose.
#
# The gripper approach axis is EE-frame +Y (the Robotiq fingers extend along the
# wrist_3 flange normal). At this pose that axis = world (0, 0, -1) EXACTLY —
# verified via forward_kinematics_full. With the UR base upright at world
# ≈ (0.027, -0.874, 1.028) facing +world-X and the Robotiq tool offset (0.2422 m
# along EE local Y), the EE lands at world ≈ (0.419, -0.710, 1.156) — ~13 cm
# above the table top, in front of the base, tool pointing dead down. Jacobian is
# well-conditioned (cond ≈ 3.2, far from any singularity).
#
# Joint angles (deg): pan 0, lift -107.39, elbow 144.04, wrist_1 -126.65,
# wrist_2 -90, wrist_3 0. The adapter FORCES these on startup (post_reset) — the
# USD initial-state authoring is unreliable, so without that the arm would hold
# its loaded zero pose (gripper horizontal). set_park_pose.py mirrors them too.
HOME_POSITION = np.radians(np.array([
    0.0,        # shoulder_pan
    -107.39,    # shoulder_lift
    144.04,     # elbow
    -126.65,    # wrist_1
    -90.0,      # wrist_2
    0.0,        # wrist_3
]))


# ── Rigid-body helpers ──────────────────────────────────────────────

def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ])


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


def _rot_axis(axis: tuple, angle: float) -> np.ndarray:
    """4×4 homogeneous rotation by *angle* about a local unit *axis*.

    The UR joints rotate about mixed local axes (Z or Y), so FK cannot
    assume the local Z used by the Lite6/Mecharm.  Uses Rodrigues' formula.
    """
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.eye(4)
    ax = ax / n
    x, y, z = ax
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _translate(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/ROS convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _joint_frame(xyz: tuple, rpy: tuple) -> np.ndarray:
    """Fixed transform between parent and child frames (before rotation)."""
    return _translate(*xyz) @ _rpy_matrix(*rpy)


# ── Forward kinematics ─────────────────────────────────────────────

def forward_kinematics(
    q: np.ndarray, tool_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute end-effector world position for joint angles *q* (6,).

    If *tool_offset* (3,) is given it is applied as a fixed translation
    beyond the last joint frame (e.g. to account for a mounted gripper).

    Returns (3,) position vector [x, y, z].
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        p = _JOINT_PARAMS[i]
        T = T @ _joint_frame(p["xyz"], p["rpy"]) @ _rot_axis(p["axis"], q[i])
    if tool_offset is not None:
        T = T @ _translate(tool_offset[0], tool_offset[1], tool_offset[2])
    return T[:3, 3].copy()


def forward_kinematics_full(
    q: np.ndarray, tool_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute end-effector 4×4 homogeneous transform for joint angles *q*.

    Returns the full transform matrix (position + orientation).
    """
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        p = _JOINT_PARAMS[i]
        T = T @ _joint_frame(p["xyz"], p["rpy"]) @ _rot_axis(p["axis"], q[i])
    if tool_offset is not None:
        T = T @ _translate(tool_offset[0], tool_offset[1], tool_offset[2])
    return T.copy()


def forward_kinematics_all(q: np.ndarray) -> list:
    """
    Compute transforms for every joint frame (useful for visualisation).

    Returns a list of 7 4×4 matrices: [T_base, T_after_j1, …, T_after_j6].
    """
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        p = _JOINT_PARAMS[i]
        T = T @ _joint_frame(p["xyz"], p["rpy"]) @ _rot_axis(p["axis"], q[i])
        frames.append(T.copy())
    return frames


# ── Joint axes ──────────────────────────────────────────────────────

def compute_joint_axes(q: np.ndarray) -> np.ndarray:
    """Compute the rotation axis of each joint in world frame.

    For each revolute joint the world axis is the local joint axis mapped
    through the frame just before that joint's own rotation is applied.

    Returns (6, 3) array: row *i* is the unit rotation axis of joint *i*.
    """
    T = np.eye(4)
    axes = np.zeros((NUM_JOINTS, 3))
    for i in range(NUM_JOINTS):
        p = _JOINT_PARAMS[i]
        T_pre = T @ _joint_frame(p["xyz"], p["rpy"])
        axes[i] = T_pre[:3, :3] @ np.asarray(p["axis"], dtype=float)
        T = T_pre @ _rot_axis(p["axis"], q[i])
    return axes


# ── Jacobian ────────────────────────────────────────────────────────

def compute_jacobian(
    q: np.ndarray, tool_offset: Optional[np.ndarray] = None, eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute the 3×6 position Jacobian numerically via finite differences.

    J[:, i] ≈ ∂FK/∂q_i
    """
    p0 = forward_kinematics(q, tool_offset)
    J = np.zeros((3, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        q_pert = q.copy()
        q_pert[i] += eps
        p_pert = forward_kinematics(q_pert, tool_offset)
        J[:, i] = (p_pert - p0) / eps
    return J


# ── Differential IK solver (hand-rolled DLS) ───────────────────────

def solve_ik_step(
    q: np.ndarray,
    target_delta: np.ndarray,
    damping: float = 0.05,
    max_step: float = 0.1,
    tool_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute a joint-angle increment to move the end-effector by *target_delta*.

    Uses damped least-squares on the position Jacobian:
        Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ Δx

    Args:
        q:            Current joint angles (6,).
        target_delta: Desired end-effector displacement (3,) in world frame.
        damping:      Damping factor λ for singularity robustness.
        max_step:     Maximum joint-angle step (rad) per call.
        tool_offset:  Optional fixed translation beyond the last joint.

    Returns:
        dq: Joint angle increment (6,), clamped to ±max_step.
    """
    J = compute_jacobian(q, tool_offset)
    JT = J.T
    lam2 = damping ** 2
    dq = JT @ np.linalg.solve(J @ JT + lam2 * np.eye(3), target_delta)
    dq = np.clip(dq, -max_step, max_step)
    return dq


def solve_ik_pose_step(
    q: np.ndarray,
    pos_delta: np.ndarray,
    rot_delta: np.ndarray,
    damping: float = 0.05,
    max_step: float = 0.1,
    tool_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Full 6-DOF damped-least-squares step for a combined position + orientation
    velocity command.

    Builds a 6×6 spatial Jacobian (3 position rows from finite differences,
    3 angular rows from the joint axes) and solves the damped least-squares
    system for the joint increment:
        Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ [Δp; Δθ]

    Args:
        q:           Current joint angles (6,).
        pos_delta:   Desired EE translation (3,) in world frame.
        rot_delta:   Desired EE rotation as a rotation-vector (3,) in world frame.
        damping:     DLS damping factor λ.
        max_step:    Maximum joint-angle step (rad) per call.
        tool_offset: Optional fixed translation beyond the last joint.

    Returns:
        dq: Joint angle increment (6,), clamped to ±max_step.
    """
    Jp = compute_jacobian(q, tool_offset)          # (3, 6)
    Jw = compute_joint_axes(q).T                    # (3, 6) angular Jacobian
    J = np.vstack([Jp, Jw])                         # (6, 6)
    dx = np.concatenate([pos_delta, rot_delta])     # (6,)
    JT = J.T
    lam2 = damping ** 2
    dq = JT @ np.linalg.solve(J @ JT + lam2 * np.eye(6), dx)
    dq = np.clip(dq, -max_step, max_step)
    return dq


def solve_ik_position(
    target_pos: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-3,
    damping: float = 0.05,
    step_scale: float = 1.0,
    tool_offset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, bool]:
    """
    Solve inverse kinematics for a target end-effector position.

    Iterates damped-least-squares steps until the position error is below
    *tol* or *max_iter* is reached.

    Returns:
        (q_solution, final_error, converged)
    """
    q = q_init.copy() if q_init is not None else HOME_POSITION.copy()

    for _ in range(max_iter):
        ee = forward_kinematics(q, tool_offset)
        error = target_pos - ee
        err_norm = np.linalg.norm(error)
        if err_norm < tol:
            return q, err_norm, True

        dq = solve_ik_step(q, error * step_scale, damping=damping, tool_offset=tool_offset)
        q = q + dq
        q = np.clip(q, JOINT_LOWER, JOINT_UPPER)

    ee = forward_kinematics(q, tool_offset)
    err_norm = float(np.linalg.norm(target_pos - ee))
    return q, err_norm, err_norm < tol


# ── UR10Controller (wraps kinematics for a single arm) ─────────────

class UR10Controller:
    """
    High-level controller for one UR10 / UR10e inside Isaac Sim.

    Wraps the FK/IK solver and provides velocity-mode end-effector control
    suitable for real-time analog-stick teleoperation.  Mirrors the
    MecharmController / Lite6Controller interface so a shared sort layer can
    drive all three arms uniformly.
    """

    def __init__(
        self,
        name: str = "ur10e",
        damping: float = 0.05,
        max_ee_speed: float = 0.15,      # m/s
        max_joint_step: float = 0.05,     # rad per control tick
        tool_offset: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.damping = damping
        self.max_ee_speed = max_ee_speed
        self.max_joint_step = max_joint_step
        self.tool_offset = tool_offset

        # Current joint state (maintained by the controller; synced from sim)
        self._q = HOME_POSITION.copy()

    @property
    def joint_positions(self) -> np.ndarray:
        return self._q.copy()

    @joint_positions.setter
    def joint_positions(self, q: np.ndarray) -> None:
        self._q = np.clip(q, JOINT_LOWER, JOINT_UPPER)

    @property
    def ee_position(self) -> np.ndarray:
        """Current end-effector position from FK."""
        return forward_kinematics(self._q, self.tool_offset)

    def apply_ee_velocity(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Move end-effector by velocity (3,) for time-step *dt*.

        Computes the joint-angle increment via differential IK and returns
        the new clamped joint positions.
        """
        speed = np.linalg.norm(velocity)
        if speed > self.max_ee_speed:
            velocity = velocity * (self.max_ee_speed / speed)

        dx = velocity * dt
        dq = solve_ik_step(
            self._q, dx, damping=self.damping,
            max_step=self.max_joint_step, tool_offset=self.tool_offset,
        )
        self._q = np.clip(self._q + dq, JOINT_LOWER, JOINT_UPPER)
        return self._q.copy()

    def go_home(self) -> np.ndarray:
        """Reset joints to home position."""
        self._q = HOME_POSITION.copy()
        return self._q.copy()

    def get_workspace_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Rough estimate of the reachable workspace bounding box."""
        rng = np.random.default_rng(42)
        positions = []
        for _ in range(5000):
            q = rng.uniform(JOINT_LOWER, JOINT_UPPER)
            positions.append(forward_kinematics(q))
        positions = np.array(positions)
        return positions.min(axis=0), positions.max(axis=0)
