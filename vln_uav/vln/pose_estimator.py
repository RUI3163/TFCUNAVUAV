"""Stage 4 (self-localisation): GPS-denied dead reckoning.

  nadir optical flow (Lucas-Kanade, median of tracked corners)
  x altitude (metres/pixel scale)  +  IMU/attitude yaw  ->  (north, east) in the R frame.

Known limits (also documented in the README): needs textured ground; ignores roll/pitch
de-rotation (fine for slow, level lawn-mower legs); drift grows ~ linearly with distance.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraCfg, NavCfg
from .geometry import body_to_ned, wrap_pi


@dataclass(frozen=True)
class PoseEstimate:
    n: float            # north of R (m)
    e: float            # east of R (m)
    alt: float          # altitude above ground (m)
    yaw: float          # rad, clockwise from north
    dist: float         # cumulative distance flown (m) -> drift model input
    flow_ok: bool       # last flow update had enough features


class DeadReckoner:
    def __init__(self, cam: CameraCfg, nav: NavCfg, start_ne: Tuple[float, float]):
        self.cam, self.nav = cam, nav
        self._lock = threading.RLock()
        self._n, self._e = start_ne
        self._alt = 0.0
        self._yaw = 0.0
        self._yaw_init = False
        self._dist = 0.0
        self._flow_ok = False
        self._prev: Optional[np.ndarray] = None
        self._last_imu_t: Optional[float] = None
        self.lost_frames = 0

    # ---- sensor inputs -----------------------------------------------------
    def on_imu(self, wz: float, t: float) -> None:
        with self._lock:
            if self._last_imu_t is not None and self.nav.yaw_source == "gyro" and self._yaw_init:
                dt = min(max(t - self._last_imu_t, 0.0), 0.1)
                self._yaw = wrap_pi(self._yaw + self.nav.gyro_yaw_sign * wz * dt)
            self._last_imu_t = t

    def on_pose(self, altitude: float, yaw_pose: float, _wn: float, _we: float) -> None:
        """Altitude (rangefinder/baro stand-in) always; yaw only for initial alignment or yaw_source='pose'.
        World x/y from the sim are deliberately ignored -> GPS-denied."""
        with self._lock:
            self._alt = altitude
            if self.nav.yaw_source == "pose" or not self._yaw_init:
                self._yaw = yaw_pose
                self._yaw_init = True

    # ---- optical flow ------------------------------------------------------
    def update_frame(self, bgr: np.ndarray) -> PoseEstimate:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        with self._lock:
            alt, yaw = self._alt, self._yaw
        if self._prev is not None and alt > 0.5:
            self._integrate_flow(self._prev, gray, alt, yaw)
        self._prev = gray
        return self.state()

    def _integrate_flow(self, prev: np.ndarray, cur: np.ndarray, alt: float, yaw: float) -> None:
        pts = cv2.goodFeaturesToTrack(prev, self.nav.flow_max_corners, 0.01, 8)
        ok = False
        if pts is not None and len(pts) >= self.nav.flow_min_features:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = st.reshape(-1) == 1
            if good.sum() >= self.nav.flow_min_features:
                d = (nxt - pts).reshape(-1, 2)[good]
                du, dv = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
                # Features move opposite to the camera: camera forward -> scene moves toward image bottom (+v);
                # camera right -> scene moves toward image left (-u).
                fwd = dv * alt / self.cam.fy
                right = -du * alt / self.cam.fx
                dn, de = body_to_ned(fwd, right, yaw)
                with self._lock:
                    self._n += dn
                    self._e += de
                    self._dist += math.hypot(dn, de)
                ok = True
        with self._lock:
            self._flow_ok = ok
            self.lost_frames = 0 if ok else self.lost_frames + 1

    def state(self) -> PoseEstimate:
        with self._lock:
            return PoseEstimate(self._n, self._e, self._alt, self._yaw, self._dist, self._flow_ok)
