"""
Observation capture — top-down + eye-in-hand RGB/depth.

The supervised scorer conditions on the PRE-GRASP observation (the arm at the
descend-arrival pose, jaws still open). This module owns two cameras:

  • top-down  — fixed above the work zone, looking straight down
  • eye-in-hand — parented under the gripper, looking along the approach axis

The Isaac camera sensor API has shifted across 5.x/6.x; calls here are defensive
(getattr fallbacks) and return None on failure rather than crashing a long
data-gen run. Validate the exact methods on the GPU box (Step 0).

Import only after a SimulationApp exists.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Observer:
    def __init__(self, tool_link_path: str, work_centre, table_top_z: float,
                 resolution=(320, 240)):
        self.resolution = resolution
        self.tool_link_path = tool_link_path
        self.work_centre = work_centre
        self.table_top_z = table_top_z
        self._top = None
        self._eih = None

    def setup(self):
        from isaacsim.sensors.camera import Camera
        cx, cy = self.work_centre
        # top-down camera ~0.5 m above the work centre, looking down (-Z).
        self._top = Camera(
            prim_path="/World/TopCam",
            position=np.array([cx, cy, self.table_top_z + 0.5]),
            frequency=20, resolution=self.resolution,
            orientation=_euler_quat(0.0, 90.0, 0.0),   # pitch down to look at -Z
        )
        # eye-in-hand camera under the gripper, looking along the approach axis (+Y).
        self._eih = Camera(
            prim_path=self.tool_link_path + "/gripper/EyeInHand",
            translation=np.array([0.0, 0.04, 0.06]),
            frequency=20, resolution=self.resolution,
        )
        for cam in (self._top, self._eih):
            try:
                cam.initialize()
                if hasattr(cam, "add_distance_to_image_plane_to_frame"):
                    cam.add_distance_to_image_plane_to_frame()
            except Exception:
                pass

    def _rgb(self, cam) -> Optional[np.ndarray]:
        if cam is None:
            return None
        try:
            rgba = cam.get_rgba()
            if rgba is None or rgba.size == 0:
                return None
            return np.asarray(rgba)[:, :, :3].astype(np.uint8)
        except Exception:
            return None

    def _depth(self, cam) -> Optional[np.ndarray]:
        if cam is None:
            return None
        try:
            if hasattr(cam, "get_depth"):
                d = cam.get_depth()
                if d is not None and getattr(d, "size", 0):
                    return np.asarray(d, dtype=np.float32)
            frame = cam.get_current_frame()
            d = frame.get("distance_to_image_plane")
            return np.asarray(d, dtype=np.float32) if d is not None else None
        except Exception:
            return None

    def capture(self) -> dict:
        """Return {topdown_rgb, topdown_depth, eih_rgb, eih_depth} arrays (or None)."""
        return {
            "topdown_rgb": self._rgb(self._top),
            "topdown_depth": self._depth(self._top),
            "eih_rgb": self._rgb(self._eih),
            "eih_depth": self._depth(self._eih),
        }


def _euler_quat(roll_deg, pitch_deg, yaw_deg):
    """(w, x, y, z) from XYZ Euler degrees — small helper for camera orientation."""
    import math
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
