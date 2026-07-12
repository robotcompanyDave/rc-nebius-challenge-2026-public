"""Procedural soft work-surface deformation — NO physics solver.

A height-field "pad" that dents where a presser (the gripper fingertip) pushes into it and
springs back when the presser lifts. The deformation is computed analytically, so it is 100%
reliable — the opposite of the PhysX volume-deformable path, which (see the fem-pad spike in
rc-rp-spikes) cannot be loaded by a rigid object in this Isaac build.

Vendored from rc-rp-spikes/spikes/fem-pad/softpad.py (runtime subset only — the weight<->
geometry analysis helpers used to build the spike's charts are dropped). numpy-only.

Model (per presser foot, a flat-bottomed punch):
  * UNDER the foot  -> the surface conforms to the foot: depressed by the penetration `pen`.
  * AROUND the foot -> a smooth cosine falloff from `pen` at the foot edge to 0 at distance
                       `spread` beyond it. `spread` grows with `pen` (press deeper -> the
                       dent reaches further), like real foam.
  * Multiple feet   -> the surface takes the deepest depression at each point.
  * Reformation     -> each tick the field relaxes toward the current target, so when the
                       feet lift the dent eases back to flat (reform_rate 1.0 = instant,
                       <1 = visible spring-back).

In the UR10e adapter: build a tessellated grid mesh for the sort surface, then each sim step
call target_field(...) with the fingertip feet, relax() toward it, and write the heights to
the mesh points (purely visual — the physical give is the collision filter + the IK floor +
the rigid hard-stop collider, not this skin).
"""
from __future__ import annotations
import numpy as np


class SoftPad:
    def __init__(self, spread_base_m: float = 0.002, spread_gain: float = 1.5,
                 max_indent_m: float = 0.010, reform_rate: float = 0.25):
        self.spread_base = float(spread_base_m)      # lateral reach at zero depth
        self.spread_gain = float(spread_gain)        # extra reach per metre of depth
        self.max_indent = float(max_indent_m)        # clamp (can't sink past the pad)
        self.reform_rate = float(reform_rate)        # 1.0 = instant, <1 = eased spring-back

    def spread_of(self, pen_m: float) -> float:
        """Lateral reach (m) of the dent beyond the foot edge, for a given penetration."""
        return self.spread_base + self.spread_gain * pen_m

    def target_field(self, X: np.ndarray, Y: np.ndarray, rest_z: float, feet) -> np.ndarray:
        """Deformed height at each (X, Y). feet: iterable of (px, py, half_w, pen) in metres.
        Square feet (Chebyshev distance) to match block/fingertip footprints."""
        Z = np.full_like(X, rest_z, dtype=float)
        for (px, py, hw, pen) in feet:
            pen = min(float(pen), self.max_indent)
            if pen <= 0:
                continue
            d = np.maximum(np.abs(X - px), np.abs(Y - py))   # distance to foot centre (square)
            edge = np.maximum(0.0, d - hw)                   # distance beyond the foot edge
            R = self.spread_of(pen)
            depress = np.where(
                edge <= 0.0, pen,                                        # under the foot
                np.where(edge < R, pen * 0.5 * (1.0 + np.cos(np.pi * edge / R)),  # falloff
                         0.0))
            Z = np.minimum(Z, rest_z - depress)              # deepest wins
        return Z

    def relax(self, Z_cur: np.ndarray, Z_target: np.ndarray) -> np.ndarray:
        """One reformation tick: ease the current surface toward the target."""
        return Z_cur + (Z_target - Z_cur) * self.reform_rate
