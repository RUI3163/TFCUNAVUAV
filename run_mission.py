"""Entry point:  python run_mission.py [--spec-json mission_spec.json] [--nav-mode estimate|groundtruth]

Geometry overrides (all optional; defaults live in vln/config.py):
  --r-world N E         Reference Point R in world NED metres        (default: Car_65 - (12, 26))
  --area-center N E     centre of the 40x40 m search square, R frame (default: 20 20 => R is its SW corner)
  --start-offset N E    UAV start in the R frame; disables auto-detection from the live spawn pose
A simulator-free preflight check runs first and aborts (exit code 2) if Car_65 would not be swept.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime

from vln.config import Config
from vln.pipeline import MissionRunner
from vln.stage2_mission import MissionSpec
from vln.stage3_planner import coverage_margin, plan_search

log = logging.getLogger("run_mission")


def preflight_geometry(cfg: Config) -> bool:
    """Static check (no simulator): Car_65's configured world pose, mapped into the R frame, must lie inside the
    search square AND be swept by the planned lawn-mower footprint. Evaluation data only -- the mission itself
    never sees the target position."""
    g = cfg.geometry
    car_r = g.gt_object_in_r()
    n0, e0, n1, e1 = g.area_bounds()
    log.info("R (world NED): N %.3f, E %.3f | search area (R frame): N [%.1f, %.1f] E [%.1f, %.1f]",
             g.r_world_ne[0], g.r_world_ne[1], n0, n1, e0, e1)
    log.info("UAV start (R frame, %s): N %.1f, E %.1f",
             "static fallback; live spawn overrides" if g.auto_start_from_sim else "fixed", *g.start_offset_ne)
    if car_r is None:
        log.warning("No gt_object_world_ned configured -> skipping coverage preflight.")
        return True
    inside = g.in_area(car_r)
    log.info("%s in R frame: N %.2f, E %.2f -> %s the %.0f x %.0f m area",
             g.gt_object_name, car_r[0], car_r[1], "INSIDE" if inside else "OUTSIDE", g.area_size_m, g.area_size_m)
    if not inside:
        log.error("%s lies outside the search area. Fix r_world_ne / area_center_ne.", g.gt_object_name)
        return False
    margin = coverage_margin(plan_search(g, cfg.camera), g.start_offset_ne, car_r, g.altitude_m, cfg.camera)
    if margin < 0:
        log.error("The planned sweep does not cover %s.", g.gt_object_name)
        return False
    log.info("Planned sweep covers %s with >= %.1f m position-error margin.", g.gt_object_name, margin)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="GPS-denied VLN UAV mission on Project AirSim")
    ap.add_argument("--scene", default=None, help="scene config name (overrides SimCfg.scene_config)")
    ap.add_argument("--sim-config-path", default=None)
    ap.add_argument("--spec-json", default=None, help="skip Stage 2 and load a saved MissionSpec JSON")
    ap.add_argument("--nav-mode", choices=["estimate", "groundtruth"], default=None)
    ap.add_argument("--r-world", type=float, nargs=2, metavar=("N", "E"), default=None,
                    help="Reference Point R in world NED metres")
    ap.add_argument("--area-center", type=float, nargs=2, metavar=("N", "E"), default=None,
                    help="centre of the search square in the R frame")
    ap.add_argument("--start-offset", type=float, nargs=2, metavar=("N", "E"), default=None,
                    help="UAV start in the R frame (disables auto-detection from the live spawn pose)")
    ap.add_argument("--det-weights", default=None, help="detector weights, e.g. runs/detect/train-3/weights/best.pt")
    ap.add_argument("--det-classes", type=int, nargs="+", default=None, help="class ids to keep (fine-tuned single class: 0)")
    ap.add_argument("--imgsz", type=int, default=None, help="detector input size (debug run used 640)")
    ap.add_argument("--det-conf", type=float, default=None)
    ap.add_argument("--verifier-mode", choices=["color", "yesno"], default=None)
    ap.add_argument("--debug-dir", default=None, help="save annotated Tier-1 frames here (diagnostics)")
    ap.add_argument("--skip-preflight", action="store_true", help="fly even if the preflight geometry check fails")
    args = ap.parse_args()

    cfg = Config()
    if args.scene:
        cfg.sim.scene_config = args.scene
    if args.sim_config_path:
        cfg.sim.sim_config_path = args.sim_config_path
    if args.nav_mode:
        cfg.nav.nav_mode = args.nav_mode
    if args.det_weights:
        cfg.perception.detector_weights = args.det_weights
    if args.det_classes:
        cfg.perception.det_classes = tuple(args.det_classes)
    if args.imgsz:
        cfg.perception.imgsz = args.imgsz
    if args.det_conf is not None:
        cfg.perception.det_conf = args.det_conf
    if args.verifier_mode:
        cfg.verifier.mode = args.verifier_mode
    if args.debug_dir:
        cfg.perception.debug_dir = args.debug_dir
    if args.r_world:
        cfg.geometry.r_world_ne = tuple(args.r_world)
    if args.area_center:
        cfg.geometry.area_center_ne = tuple(args.area_center)
    if args.start_offset:
        cfg.geometry.start_offset_ne = tuple(args.start_offset)
        cfg.geometry.auto_start_from_sim = False

    os.makedirs(cfg.logs_dir, exist_ok=True)
    log_path = os.path.join(cfg.logs_dir, f"mission_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])

    if not preflight_geometry(cfg) and not args.skip_preflight:
        return 2

    spec = MissionSpec.model_validate_json(open(args.spec_json).read()) if args.spec_json else None
    report = asyncio.run(MissionRunner(cfg, spec).run())
    return 0 if report["found"] else 1


if __name__ == "__main__":
    sys.exit(main())
