"""Stage 4 (self-localisation): GPS-denied dead reckoning.

  nadir optical flow (Lucas-Kanade, median of tracked corners)
  x altitude (metres/pixel scale)  +  IMU/attitude yaw  ->  (north, east) in the R frame.

Known limits (also documented in the README): needs textured ground; ignores roll/pitch
de-rotation (fine for slow, level lawn-mower legs); drift grows ~ linearly with distance.
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraCfg, NavCfg
from .geometry import body_to_ned, wrap_pi

log = logging.getLogger(__name__)


def perspective_to_planar(depth: np.ndarray, cam: CameraCfg) -> np.ndarray:
    """Range along each pixel ray -> distance along the optical axis (what flow scaling needs)."""
    h, w = depth.shape[:2]
    u = np.arange(w, dtype=np.float32) - cam.cx
    v = np.arange(h, dtype=np.float32) - cam.cy
    r2 = u[None, :] ** 2 + v[:, None] ** 2
    return depth * (cam.fx / np.sqrt(cam.fx ** 2 + r2))


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
        self._prev_depth: Optional[np.ndarray] = None
        self._depth_checked = False
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
    def update_frame(self, bgr: np.ndarray, depth: Optional[np.ndarray] = None) -> PoseEstimate:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        planar = self._prep_depth(depth, gray.shape) if (depth is not None and self.nav.use_depth_scale) else None
        with self._lock:
            alt, yaw = self._alt, self._yaw
        if planar is not None and not self._depth_checked and alt > 5.0:
            self._depth_checked = True
            h, w = depth.shape[:2]
            c = np.nanmedian(depth[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8])
            k = np.nanmedian(np.concatenate([depth[:12, :12].ravel(), depth[:12, -12:].ravel(),
                                             depth[-12:, :12].ravel(), depth[-12:, -12:].ravel()]))
            log.info("Depth check at alt %.1f m: centre %.1f m, corners %.1f m (corner/centre %.2f; ~1.5 over flat "
                     "ground = PERSPECTIVE, ~1.0 = PLANAR; CameraCfg.depth_is_perspective=%s)",
                     alt, c, k, k / c if c else float("nan"), self.cam.depth_is_perspective)
        if self._prev is not None and alt > 0.5:
            self._integrate_flow(self._prev, gray, alt, yaw, self._prev_depth)
        self._prev, self._prev_depth = gray, planar
        return self.state()

    def _prep_depth(self, depth: np.ndarray, shape) -> np.ndarray:
        d = depth
        if d.shape[:2] != shape[:2]:
            d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return perspective_to_planar(d, self.cam) if self.cam.depth_is_perspective else d

    def _integrate_flow(self, prev: np.ndarray, cur: np.ndarray, alt: float, yaw: float,
                        prev_depth: Optional[np.ndarray] = None) -> None:
        pts = cv2.goodFeaturesToTrack(prev, self.nav.flow_max_corners, 0.01, 8)
        ok = False
        if pts is not None and len(pts) >= self.nav.flow_min_features:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
            good = st.reshape(-1) == 1
            if good.sum() >= self.nav.flow_min_features:
                d = (nxt - pts).reshape(-1, 2)[good]
                p0 = pts.reshape(-1, 2)[good]
                # Ground displacement per feature = pixel flow * (depth of THAT feature) / f.
                # Features move opposite to the camera: camera forward -> scene moves toward image bottom (+v);
                # camera right -> scene moves toward image left (-u).
                z = np.full(len(d), alt, dtype=np.float32)
                if prev_depth is not None:
                    ui = np.clip(p0[:, 0].round().astype(int), 0, prev_depth.shape[1] - 1)
                    vi = np.clip(p0[:, 1].round().astype(int), 0, prev_depth.shape[0] - 1)
                    zd = prev_depth[vi, ui]
                    valid = np.isfinite(zd) & (zd > 0.5) & (zd < 4.0 * max(alt, 5.0))
                    if valid.sum() >= self.nav.flow_min_features:
                        z, d = zd[valid], d[valid]
                fwd = float(np.median(d[:, 1] * z / self.cam.fy))
                right = float(np.median(-d[:, 0] * z / self.cam.fx))
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
