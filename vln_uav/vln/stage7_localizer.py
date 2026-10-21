"""Stage 7: target localisation -- nadir pinhole projection + N-consecutive-agreement gate."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CameraCfg, LocalizationCfg, VerifierCfg
from .geometry import pixel_to_ground_offset
from .stage6_verifier import VerifyResult

log = logging.getLogger(__name__)


@dataclass
class TargetFix:
    n: float                 # north of R (m)
    e: float                 # east of R (m)
    var_indep: float         # per-fix variance that averages down over N (pixel + altitude + yaw)
    drift: float             # dead-reckoning drift sigma (common-mode; does not average down)
    frame_idx: int


@dataclass
class FinalFix:
    n: float
    e: float
    sigma_m: float
    fixes: List[TargetFix]
    n_frames: int


class Localizer:
    def __init__(self, cam: CameraCfg, cfg: LocalizationCfg):
        self.cam, self.cfg = cam, cfg

    def localize(self, res: VerifyResult) -> TargetFix:
        c, est = res.cand, res.cand.est
        x1, y1, x2, y2 = c.box
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dn, de = pixel_to_ground_offset(u, v, est.alt, est.yaw, self.cam)
        r = math.hypot(dn, de)                                   # ground range from nadir point
        s = self.cfg
        pix = (s.pixel_sigma_px / self.cam.fx * est.alt) ** 2
        alt = (r / max(est.alt, 1e-3) * s.altitude_sigma_m) ** 2   # range scales linearly with altitude
        yaw = (r * s.yaw_sigma_rad) ** 2
        return TargetFix(est.n + dn, est.e + de, pix + alt + yaw, s.drift_frac * est.dist, c.frame_idx)


class ConsensusGate:
    """Emits a FinalFix once N *consecutive* verification results are positive AND spatially agree.

    Any negative verdict resets the streak (conservative: favours precision). A positive that disagrees
    with the running streak by more than `agree_radius_m` restarts the streak with that fix.
    """

    def __init__(self, localizer: Localizer, cfg: VerifierCfg):
        self.loc, self.cfg = localizer, cfg
        self.streak: List[TargetFix] = []

    def add(self, res: VerifyResult) -> Optional[FinalFix]:
        if not res.is_target:
            if self.streak:
                log.info("Consensus reset by negative verdict (streak was %d).", len(self.streak))
            self.streak = []
            return None
        fix = self.loc.localize(res)
        if self.streak:
            mn = float(np.mean([f.n for f in self.streak]))
            me = float(np.mean([f.e for f in self.streak]))
            if math.hypot(fix.n - mn, fix.e - me) > self.cfg.agree_radius_m:
                log.info("Positive verdict disagrees with streak (%.1f m); restarting streak.",
                         math.hypot(fix.n - mn, fix.e - me))
                self.streak = []
        self.streak.append(fix)
        log.info("Consensus %d/%d at (N %.2f, E %.2f)", len(self.streak), self.cfg.consensus_n, fix.n, fix.e)
        if len(self.streak) < self.cfg.consensus_n:
            return None
        return self._fuse(self.streak[-self.cfg.consensus_n:])

    @staticmethod
    def _fuse(fixes: List[TargetFix]) -> FinalFix:
        n = np.array([f.n for f in fixes])
        e = np.array([f.e for f in fixes])
        N = len(fixes)
        spread2 = (n.var(ddof=1) + e.var(ddof=1)) / 2.0 if N > 1 else 0.0   # empirical scatter per axis
        indep = float(np.mean([f.var_indep for f in fixes])) / N
        drift = float(np.mean([f.drift for f in fixes]))
        sigma = math.sqrt(indep + spread2 / N + drift ** 2)
        return FinalFix(float(n.mean()), float(e.mean()), sigma, list(fixes), N)
