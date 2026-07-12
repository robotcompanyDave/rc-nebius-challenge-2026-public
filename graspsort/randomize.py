"""
Domain randomization for grasp-attempt data generation.

Sampled per attempt:
  • the SCENE — which parts (kind/size/pose/yaw) sit on the platform and where:
    a single part, a loose scatter, or a touching BUNDLE (the clutter the
    drag-separate strategy exists for);
  • the ACTION — the candidate grasp to execute and LABEL: xy/yaw/depth
    perturbations plus the pick STRATEGY (direct / tilt / lip-press) and an
    optional pre-drag. The spread of successes and failures across these is
    what the scorer learns from.

Pure-python (random + numpy); produces `parts.PartSpec`s and a candidate dict
the controller consumes. Optional nuisance knobs (friction/light/camera) are
returned for the env/observer to apply.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

from .parts import PartSpec, POSE_CLASSES, KINDS, WORK_SIZES, bundle_xy, scatter_xy

# The working span is M12 down to M6; m12 stays the most-visited size.
_SIZE_WEIGHTS = [("m12", 0.4), ("m10", 0.2), ("m8", 0.2), ("m6", 0.2)]
_KIND_WEIGHTS = [("nut", 0.4), ("bolt", 0.3), ("washer", 0.3)]
_STRATEGY_WEIGHTS = [("direct", 0.45), ("tilt", 0.3), ("lip", 0.25)]


def _weighted(rng: random.Random, weighted):
    r = rng.random()
    acc = 0.0
    for val, w in weighted:
        acc += w
        if r <= acc:
            return val
    return weighted[-1][0]


def sample_scene(rng: random.Random, work_centre, spread: float = 0.03) -> PartSpec:
    """Single-part scene (kept for compatibility with the v1 dataset jobs)."""
    kind = _weighted(rng, _KIND_WEIGHTS)
    size = _weighted(rng, _SIZE_WEIGHTS)
    pose = rng.choice(POSE_CLASSES)
    cx, cy = work_centre
    xy = (cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread))
    return PartSpec(kind=kind, size=size, pose=pose, xy=xy,
                    rotz_deg=rng.uniform(0.0, 360.0))


def sample_scene_multi(rng: random.Random, work_centre,
                       n_parts: Optional[int] = None) -> List[PartSpec]:
    """Multi-part scene: 1–5 parts, ~40% as a touching BUNDLE, else a loose
    scatter. Thin parts lie mostly flat (the realistic dump)."""
    n = n_parts if n_parts is not None else rng.choice([1, 2, 3, 3, 4, 5])
    bundle = n >= 2 and rng.random() < 0.4
    xys = (bundle_xy(work_centre, n, rng=rng)
           if bundle else scatter_xy(work_centre, n, spread=0.035,
                                     min_sep=0.03, rng=rng))
    specs = []
    for i in range(n):
        kind = _weighted(rng, _KIND_WEIGHTS)
        size = _weighted(rng, _SIZE_WEIGHTS)
        pose = ("flat" if rng.random() < 0.6 else rng.choice(POSE_CLASSES))
        specs.append(PartSpec(kind=kind, size=size, pose=pose, xy=xys[i],
                              rotz_deg=rng.uniform(0.0, 360.0)))
    return specs


def sample_candidate(rng: random.Random, kind: str,
                     heuristic_yaw: Optional[float] = None,
                     cluttered: bool = False) -> dict:
    """Candidate grasp to execute + label. Perturbs around the heuristic so the
    dataset spans good and bad grasps.

    xy_offset:   gaussian few-mm miss off the part centre
    grasp_yaw:   heuristic yaw ± perturbation (None heuristic → fully random yaw)
    grasp_dz:    depth perturbation (m), shallow/deep
    approach_dh: approach height (m)
    width:       commanded final openness target (0..1 closed)
    strategy:    direct | tilt | lip — the pick technique to try (and learn)
    tilt_deg:    leading-tip tilt for tilt/lip
    lead_dir:    None → controller picks the free side
    pre_drag:    bundle scenes sometimes drag first (separation label)
    """
    # ~60% perturb around the heuristic, ~40% wider exploration
    if rng.random() < 0.6:
        sigma_xy, sigma_yaw = 0.004, math.radians(12.0)
    else:
        sigma_xy, sigma_yaw = 0.010, math.radians(35.0)
    xy_offset = (rng.gauss(0.0, sigma_xy), rng.gauss(0.0, sigma_xy))
    if heuristic_yaw is None:
        grasp_yaw = rng.uniform(-math.pi, math.pi)
    else:
        grasp_yaw = heuristic_yaw + rng.gauss(0.0, sigma_yaw)
    strategy = _weighted(rng, _STRATEGY_WEIGHTS)
    cand = {
        "xy_offset": xy_offset,
        "grasp_yaw": grasp_yaw,
        "grasp_dz": rng.uniform(-0.003, 0.004),
        "approach_dh": rng.uniform(0.10, 0.14),
        "width": 1.0,
        "strategy": strategy,
        "tilt_deg": rng.uniform(4.0, 14.0),
        "lead_dir": None,
    }
    if cluttered and rng.random() < 0.5:
        a = rng.uniform(0.0, 2.0 * math.pi)
        cand["pre_drag"] = {"dir_xy": (math.cos(a), math.sin(a)),
                            "dist": rng.uniform(0.04, 0.09)}
    return cand


def sample_nuisance(rng: random.Random) -> dict:
    """Optional visual/physical nuisance for scorer robustness."""
    return {
        "dome_intensity": rng.uniform(900.0, 2200.0),
        "cam_jitter_m": rng.uniform(0.0, 0.01),
        "friction_scale": rng.uniform(0.85, 1.15),
    }
