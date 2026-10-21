"""Pure geometry helpers (no simulator, no torch) -> unit-testable."""
from __future__ import annotations

import math
from typing import Tuple

from .config import CameraCfg


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """NED yaw (rad, clockwise from north) from a unit quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def body_to_ned(forward: float, right: float, yaw: float) -> Tuple[float, float]:
    """Rotate a body-frame (forward, right) ground vector into (north, east)."""
    n = forward * math.cos(yaw) - right * math.sin(yaw)
    e = forward * math.sin(yaw) + right * math.cos(yaw)
    return n, e


def pixel_to_ground_offset(u: float, v: float, altitude_m: float, yaw: float,
                           cam: CameraCfg) -> Tuple[float, float]:
    """Nadir pinhole back-projection of pixel (u, v) onto flat ground.

    Returns (d_north, d_east) of the ground point relative to the point
    directly beneath the UAV.  Image-up == body-forward, image-right == body-right.
    """
    forward = (cam.cy - v) / cam.fy * altitude_m
    right = (u - cam.cx) / cam.fx * altitude_m
    return body_to_ned(forward, right, yaw)


def range_bearing(north: float, east: float) -> Tuple[float, float]:
    """(range_m, bearing_deg clockwise from north in [0, 360))."""
    return math.hypot(north, east), math.degrees(math.atan2(east, north)) % 360.0
