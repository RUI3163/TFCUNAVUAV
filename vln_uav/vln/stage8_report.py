"""Stage 8: landmark-relative mission report."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .geometry import range_bearing
from .stage2_mission import MissionSpec
from .stage7_localizer import FinalFix


def build_report(spec: MissionSpec, fix: Optional[FinalFix], stats: Dict[str, Any],
                 gt_ne_in_R: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    ref = spec.reference.id
    rep: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mission": spec.model_dump(),
        "found": fix is not None,
        "stats": stats,
    }
    if fix is None:
        rep["summary"] = f"No {spec.target.color} passenger car confirmed within the search area."
        return rep
    rng, brg = range_bearing(fix.n, fix.e)
    rep["target_relative_to_" + ref] = {
        "east_m": round(fix.e, 2), "north_m": round(fix.n, 2),
        "range_m": round(rng, 2), "bearing_deg_from_north_cw": round(brg, 1),
        "uncertainty_1sigma_m": round(fix.sigma_m, 2), "uncertainty_2sigma_m": round(2 * fix.sigma_m, 2),
        "n_consistent_verified_frames": fix.n_frames,
    }
    rep["summary"] = (f"{spec.target.color.capitalize()} passenger car located {rng:.1f} m from {ref} at bearing "
                      f"{brg:.0f}° (E {fix.e:+.1f} m, N {fix.n:+.1f} m), ±{fix.sigma_m:.1f} m (1σ).")
    if gt_ne_in_R is not None:   # offline evaluation only; never fed back into the pipeline
        err = math.hypot(fix.n - gt_ne_in_R[0], fix.e - gt_ne_in_R[1])
        rep["evaluation_ground_truth"] = {"gt_north_m": round(gt_ne_in_R[0], 2), "gt_east_m": round(gt_ne_in_R[1], 2),
                                          "position_error_m": round(err, 2)}
    return rep


def write_report(rep: Dict[str, Any], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"mission_report_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=str)
    return path
