"""
Canonical featurization for the state -> grasp-success scorer.

ONE place that turns a (part, candidate-grasp) pair into a fixed numeric feature
vector, shared by BOTH the offline trainer (jobs/train_scorer.py, reads dataset
records) and the online eval policy (jobs/eval_sort.py, scores live candidates).
Keeping it here guarantees train/serve parity — the #1 way a learned policy
silently regresses is a feature mismatch between fit and inference.

Pure-python + numpy (no Isaac / USD), so it imports anywhere.

v2 (robotiq-soft-parts): washers join the kind vocab; sizes move to the working
span m6–m12; the pick STRATEGY (direct/tilt/lip), tilt angle, pre-drag flag and
scene clutter (neighbour count / nearest distance) join the action/state. Models
trained on the v1 vocabulary are incompatible — retrain (cheap by design).

Feature vector (see FEATURE_NAMES for the exact order):
  part identity   — kind (nut/bolt/washer), size (m6..m12), pose class one-hots
  scene clutter   — n_close (neighbours within 40 mm), nearest_mm
  candidate grasp — yaw DELTA from the heuristic (sin/cos + |dyaw|), the xy miss
                    off part centre (components + magnitude), depth & approach
                    knobs, strategy one-hot, tilt_deg, pre_drag flag
The absolute world yaw is deliberately NOT a feature: it is not learnable and does
not transfer. What matters is how far the candidate deviates from the expert grasp.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

# Categorical vocabularies. Kept explicit (not inferred from data) so a model
# trained on one dataset featurizes identically on any other / at serve time.
KINDS = ("nut", "bolt", "washer")
SIZES = ("m6", "m8", "m10", "m12")      # the working span (M12 down to M6)
POSES = ("flat", "on-side", "standing", "random")
STRATEGIES = ("direct", "tilt", "lip")

FEATURE_NAMES = (
    "kind_nut", "kind_bolt", "kind_washer",
    "size_m6", "size_m8", "size_m10", "size_m12",
    "pose_flat", "pose_on-side", "pose_standing", "pose_random",
    "n_close", "nearest_mm",
    "dyaw_sin", "dyaw_cos", "dyaw_abs",
    "off_x", "off_y", "off_mag",
    "grasp_dz", "approach_dh",
    "strat_direct", "strat_tilt", "strat_lip",
    "tilt_deg", "pre_drag",
)
N_FEATURES = len(FEATURE_NAMES)


def _onehot(value, vocab) -> list:
    return [1.0 if value == v else 0.0 for v in vocab]


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def featurize(part: Dict, action: Dict, scene: Dict | None = None) -> np.ndarray:
    """(part, action[, scene]) dicts -> (N_FEATURES,) float32 vector.

    part   : {kind, size, pose, ...}
    action : {grasp_yaw, heuristic_yaw, xy_offset:[x,y], grasp_dz, approach_dh,
              strategy, tilt_deg, pre_drag}
    scene  : {n_close, nearest_mm} — clutter around the target part (0/inf-safe)
    """
    kind = part.get("kind", "nut")
    size = part.get("size", "m12")
    pose = part.get("pose", "flat")
    scene = scene or {}

    hy = action.get("heuristic_yaw")
    gy = action.get("grasp_yaw")
    dyaw = _wrap(float(gy) - float(hy)) if (hy is not None and gy is not None) else 0.0

    ox, oy = action.get("xy_offset", (0.0, 0.0))
    ox, oy = float(ox), float(oy)
    dz = float(action.get("grasp_dz", 0.0))
    adh = float(action.get("approach_dh", 0.12))
    strat = str(action.get("strategy", "direct")).lower()
    tilt = float(action.get("tilt_deg", 0.0)) if strat in ("tilt", "lip") else 0.0
    pre_drag = 1.0 if action.get("pre_drag") else 0.0

    n_close = float(scene.get("n_close", 0.0))
    nearest = scene.get("nearest_mm")
    nearest_mm = float(min(nearest, 200.0)) if nearest is not None else 200.0

    feats = (
        _onehot(kind, KINDS)
        + _onehot(size, SIZES)
        + _onehot(pose, POSES)
        + [n_close, nearest_mm]
        + [math.sin(dyaw), math.cos(dyaw), abs(dyaw)]
        + [ox, oy, math.hypot(ox, oy)]
        + [dz, adh]
        + _onehot(strat, STRATEGIES)
        + [tilt, pre_drag]
    )
    return np.asarray(feats, dtype=np.float32)


def featurize_record(rec: Dict) -> np.ndarray:
    """Featurize a dataset record (as written by logging_schema.AttemptRecord)."""
    obs = rec.get("obs", {})
    return featurize(obs.get("part", {}), rec.get("action", {}),
                     obs.get("scene", {}))
