"""
Vectorized UR10/UR10e kinematics — FK, position Jacobian and DLS-IK for a BATCH
of K arms at once, for parallel/replicated data-gen (K grasp envs in one sim).

Numerically identical to `kinematics.py` (the single-arm vendored version), just
with a leading batch axis so K arms are solved in one NumPy call instead of a
Python loop. Verified against the scalar version in tests/test_kinematics_batched.

Shapes: Q is (K, 6) joint angles; positions come back (K, 3); Jacobians (K, 3, 6).
The per-joint fixed frames and rotation axes are the SAME constants as kinematics.py
(imported, not copied), so the two can never drift apart.
"""
import numpy as np

from .kinematics import (
    _JOINT_PARAMS, _joint_frame, NUM_JOINTS, JOINT_LOWER, JOINT_UPPER,
    HOME_POSITION,
)

# ── Precompute the constants (done once at import) ─────────────────────────────
# Fixed parent→child frame before each joint's own rotation (constant, 4×4).
_F = np.stack([_joint_frame(p["xyz"], p["rpy"]) for p in _JOINT_PARAMS])   # (6,4,4)

# Per-joint cross-product matrix Kx of its (unit) rotation axis, for Rodrigues:
#   R(θ) = I + sinθ·Kx + (1-cosθ)·Kx²   (axis fixed per joint → batch only over θ)
def _skew(a):
    x, y, z = a
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

_AX = np.stack([np.asarray(p["axis"], float) / np.linalg.norm(p["axis"])
                for p in _JOINT_PARAMS])                                    # (6,3)
_KX = np.stack([_skew(a) for a in _AX])                                     # (6,3,3)
_KX2 = np.matmul(_KX, _KX)                                                  # (6,3,3)
_I3 = np.eye(3)


def _rot_axis_batch(joint_i: int, q: np.ndarray) -> np.ndarray:
    """(K,4,4) homogeneous rotations of joint `joint_i` by per-env angles q (K,)."""
    s = np.sin(q)[:, None, None]
    c = np.cos(q)[:, None, None]
    R = _I3[None] + s * _KX[joint_i][None] + (1.0 - c) * _KX2[joint_i][None]  # (K,3,3)
    T = np.zeros((q.shape[0], 4, 4))
    T[:, :3, :3] = R
    T[:, 3, 3] = 1.0
    return T


def fk_batch(Q: np.ndarray, tool_offset=None) -> np.ndarray:
    """End-effector world positions for a batch of joint configs Q (K,6) → (K,3)."""
    return fk_full_batch(Q, tool_offset)[:, :3, 3]


def fk_full_batch(Q: np.ndarray, tool_offset=None) -> np.ndarray:
    """Full EE homogeneous transforms for Q (K,6) → (K,4,4)."""
    Q = np.asarray(Q, float)
    K = Q.shape[0]
    T = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
    for i in range(NUM_JOINTS):
        T = T @ (_F[i][None] @ _rot_axis_batch(i, Q[:, i]))
    if tool_offset is not None:
        tt = np.eye(4)
        tt[:3, 3] = tool_offset
        T = T @ tt[None]
    return T


def jacobian_batch(Q: np.ndarray, tool_offset=None, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference position Jacobians for Q (K,6) → (K,3,6)."""
    Q = np.asarray(Q, float)
    p0 = fk_batch(Q, tool_offset)                       # (K,3)
    J = np.zeros((Q.shape[0], 3, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        Qp = Q.copy()
        Qp[:, i] += eps
        J[:, :, i] = (fk_batch(Qp, tool_offset) - p0) / eps
    return J


def ik_step_batch(Q, target_delta, damping=0.05, max_step=0.1, tool_offset=None):
    """One DLS position step for a batch: Q (K,6), target_delta (K,3) → dq (K,6)."""
    J = jacobian_batch(Q, tool_offset)                  # (K,3,6)
    JT = np.transpose(J, (0, 2, 1))                     # (K,6,3)
    lam2 = damping ** 2
    A = J @ JT + lam2 * _I3[None]                       # (K,3,3)
    x = np.linalg.solve(A, target_delta[..., None])     # (K,3,1)
    dq = (JT @ x)[..., 0]                               # (K,6)
    return np.clip(dq, -max_step, max_step)


def solve_ik_position_batch(target_pos, q_init=None, max_iter=200, tol=1e-3,
                            damping=0.05, step_scale=1.0, tool_offset=None):
    """Batched DLS-IK to target EE positions target_pos (K,3).

    Returns (Q (K,6), err (K,), converged (K,) bool). Envs that converge keep
    iterating harmlessly (their delta →0), so no masking is needed for correctness."""
    target_pos = np.asarray(target_pos, float)
    K = target_pos.shape[0]
    Q = (np.broadcast_to(HOME_POSITION, (K, NUM_JOINTS)).copy()
         if q_init is None else np.asarray(q_init, float).copy())
    for _ in range(max_iter):
        err = target_pos - fk_batch(Q, tool_offset)     # (K,3)
        if np.all(np.linalg.norm(err, axis=1) < tol):
            break
        Q = Q + ik_step_batch(Q, err * step_scale, damping=damping, tool_offset=tool_offset)
        Q = np.clip(Q, JOINT_LOWER, JOINT_UPPER)
    err_norm = np.linalg.norm(target_pos - fk_batch(Q, tool_offset), axis=1)
    return Q, err_norm, err_norm < tol
