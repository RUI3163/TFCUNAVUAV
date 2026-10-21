"""Stage 4 (navigation): waypoint tracking on the dead-reckoned estimate (GPS-denied closed loop)."""
from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Sequence, Tuple

from .config import Config
from .pose_estimator import DeadReckoner
from .sim_client import SimInterface

log = logging.getLogger(__name__)


class Navigator:
    def __init__(self, cfg: Config, sim: SimInterface, est: DeadReckoner):
        self.cfg, self.sim, self.est = cfg, sim, est
        # EVALUATION ONLY: dead-reckoning error vs the sim's true pose, recorded at each waypoint (never fed back).
        self.trace: list = []

    def _position(self):
        """R-frame (north, east). 'estimate' = GPS-denied dead reckoning; 'groundtruth' = sim pose (debug only)."""
        if self.cfg.nav.nav_mode == "groundtruth":
            rn, re_ = self.cfg.geometry.r_world_ne
            return self.sim.world_ne[0] - rn, self.sim.world_ne[1] - re_
        s = self.est.state()
        return s.n, s.e

    def _record(self, k: int, wn: float, we: float) -> None:
        s = self.est.state()
        tn, te = self.cfg.geometry.world_to_r(self.sim.world_ne)
        err = math.hypot(s.n - tn, s.e - te)
        self.trace.append({"waypoint": k + 1, "target_n": round(wn, 1), "target_e": round(we, 1),
                           "est_n": round(s.n, 1), "est_e": round(s.e, 1), "true_n": round(tn, 1),
                           "true_e": round(te, 1), "est_dist_m": round(s.dist, 1), "est_minus_true_m": round(err, 1)})
        log.info("WP %d: est (N %.1f, E %.1f) | true (N %.1f, E %.1f) | est-true %.1f m | est. flown %.0f m",
                 k + 1, s.n, s.e, tn, te, err, s.dist)

    def _z_world(self, altitude: float) -> float:
        return self.sim.ground_z0 - altitude

    async def takeoff_and_climb(self) -> None:
        await self.sim.takeoff()
        await self.climb_to(self.cfg.geometry.altitude_m)

    async def climb_to(self, altitude: float) -> None:
        t0 = time.monotonic()
        while abs(self.est.state().alt - altitude) > self.cfg.nav.climb_tol_m:
            await self.sim.velocity_z(0.0, 0.0, self._z_world(altitude), 0.5)
            if time.monotonic() - t0 > 60:
                raise RuntimeError("Climb to search altitude timed out.")
        log.info("At altitude %.1f m", self.est.state().alt)

    async def fly_path(self, waypoints: Sequence[Tuple[float, float]], stop: threading.Event) -> bool:
        """Fly R-frame waypoints. Returns True if completed, False if `stop` interrupted."""
        g, nv = self.cfg.geometry, self.cfg.nav
        z = self._z_world(g.altitude_m)
        for k, (wn, we) in enumerate(waypoints):
            log.info("Waypoint %d/%d -> (N %.1f, E %.1f)", k + 1, len(waypoints), wn, we)
            t0 = time.monotonic()
            while True:
                if stop.is_set():
                    await self.sim.hover()
                    return False
                cn, ce = self._position()
                dn, de = wn - cn, we - ce
                d = math.hypot(dn, de)
                if d < g.waypoint_tol_m:
                    self._record(k, wn, we)
                    break
                if time.monotonic() - t0 > 3.0 * d / g.cruise_speed_mps + 30.0:
                    log.warning("Waypoint %d timed out (est. error %.1f m); moving on.", k + 1, d)
                    self._record(k, wn, we)
                    break
                speed = min(g.cruise_speed_mps, nv.kp * d)
                await self.sim.velocity_z(dn / d * speed, de / d * speed, z, nv.control_dt_s)
                if self.est.lost_frames > 30:
                    log.warning("Optical flow lost for %d frames (textureless ground?).", self.est.lost_frames)
        await self.sim.hover()
        return True
