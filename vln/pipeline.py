"""Orchestrator: wires Stages 1-8 together."""
from __future__ import annotations

import asyncio
import logging
import math
import queue
import threading
import time
from typing import Any, Dict, Optional

from .config import Config
from .gpu import log_vram
from .pose_estimator import DeadReckoner
from .sim_client import SimInterface
from .stage2_mission import MissionSpec, parse_mission
from .stage3_planner import coverage_margin, plan_search
from .stage4_navigation import Navigator
from .stage5_perception import CropClassifier, Detector, PerceptionWorker
from .stage6_verifier import VerifierWorker, VLMVerifier, VerifyResult
from .stage7_localizer import ConsensusGate, FinalFix, Localizer
from .stage8_report import build_report, write_report

log = logging.getLogger(__name__)


class MissionRunner:
    def __init__(self, cfg: Config, spec: Optional[MissionSpec] = None):
        self.cfg, self.spec = cfg, spec

    async def run(self) -> Dict[str, Any]:
        cfg, g = self.cfg, self.cfg.geometry

        # Stage 1 + 2: fixed prompt -> validated spec (LLM loaded, used once, unloaded).
        if self.spec is None:
            self.spec = parse_mission(cfg.prompt, cfg.mission_llm)
        spec = self.spec
        if spec.target.color != "red":
            log.warning("Tier-1 classifier is trained for red_car; spec asks for '%s'.", spec.target.color)

        # Load Tier-1 + Tier-2 models BEFORE takeoff (fail early, no VRAM surprises in flight).
        detector, classifier = Detector(cfg.perception), CropClassifier(cfg.perception)
        log_vram("tier1 loaded")
        vlm = VLMVerifier(cfg.verifier, spec.target.color)

        # Simulator first: the planner needs the real spawn pose (start offset relative to R).
        sim = SimInterface(cfg)
        sim.connect()
        if g.auto_start_from_sim:
            configured = g.start_offset_ne
            actual = g.apply_spawn(sim.start_world_ne)
            log.info("Spawn world (N %.2f, E %.2f) -> start offset in R frame (N %.2f, E %.2f).",
                     sim.start_world_ne[0], sim.start_world_ne[1], actual[0], actual[1])
            if math.hypot(actual[0] - configured[0], actual[1] - configured[1]) > 1.0:
                log.warning("Start offset differs from the configured one (N %.2f, E %.2f): using the live spawn.",
                            configured[0], configured[1])

        # Stage 3: plan (R frame)
        waypoints = plan_search(g, cfg.camera)

        # Stage 4 self-localisation
        est = DeadReckoner(cfg.camera, cfg.nav, g.start_offset_ne)
        sim.add_pose_listener(est.on_pose)
        sim.add_imu_listener(est.on_imu)
        sim.start_streams()
        gt_world = None
        if g.gt_object_name:   # CONFIG SANITY CHECK ONLY -- never feeds perception/localisation
            w = sim.get_object_ne(g.gt_object_name)
            if w is None and g.gt_object_world_ned is not None:
                w = g.gt_object_world_ned[:2]
            if w is not None:
                gt_world = (float(w[0]), float(w[1]))
                n_r, e_r = g.world_to_r(w)
                inside = g.in_area((n_r, e_r))
                log.info("Config check: %s is at (N %.1f, E %.1f) in the R frame -> %s the search area.",
                         g.gt_object_name, n_r, e_r, "INSIDE" if inside else "OUTSIDE")
                if not inside:
                    log.warning("%s is outside the %.0f m search area: check GeometryCfg.r_world_ne / area_center_ne.",
                                g.gt_object_name, g.area_size_m)
                else:
                    m = coverage_margin(waypoints, g.start_offset_ne, (n_r, e_r), g.altitude_m, cfg.camera)
                    if m < 0:
                        log.warning("Planned sweep does NOT cover %s.", g.gt_object_name)
                    else:
                        log.info("Planned sweep covers %s with >= %.1f m position-error margin.", g.gt_object_name, m)
        nav = Navigator(cfg, sim, est)

        stop_threads, stop_nav = threading.Event(), threading.Event()
        results: "queue.Queue[VerifyResult]" = queue.Queue()
        verifier = VerifierWorker(cfg.verifier, vlm, results.put, stop_threads)
        perception = PerceptionWorker(cfg, sim, est, detector, classifier, verifier.submit, stop_threads)
        gate = ConsensusGate(Localizer(cfg.camera, cfg.localization), cfg.verifier)
        final: Optional[FinalFix] = None
        t_start = time.monotonic()
        min_gt_dist = [float('inf')]   # EVALUATION ONLY: closest true approach of the UAV to the target

        def track_flight() -> None:
            if gt_world is not None:
                min_gt_dist[0] = min(min_gt_dist[0], math.hypot(sim.world_ne[0] - gt_world[0], sim.world_ne[1] - gt_world[1]))

        def drain() -> Optional[FinalFix]:
            while True:
                try:
                    r = results.get_nowait()
                except queue.Empty:
                    return None
                fx = gate.add(r)
                if fx is not None:
                    return fx

        try:
            await nav.takeoff_and_climb()
            verifier.start()
            perception.start()
            flight = asyncio.create_task(nav.fly_path(waypoints, stop_nav))
            while not flight.done():
                track_flight()
                final = drain()
                if final is not None:
                    log.info("TARGET CONFIRMED -> stopping search.")
                    stop_nav.set()
                    break
                await asyncio.sleep(0.1)
            await flight
            # Sweep finished without consensus: let in-flight VLM jobs finish, then re-check.
            deadline = time.monotonic() + 15.0
            while final is None and time.monotonic() < deadline and not (verifier.idle() and results.empty()):
                final = drain()
                await asyncio.sleep(0.1)
            final = final or drain()
        finally:
            stop_threads.set()
            await asyncio.sleep(0.3)
            try:
                await sim.land()
            except Exception as exc:
                log.error("Landing failed: %s", exc)

        gt = None
        if g.gt_object_name:
            w = sim.get_object_ne(g.gt_object_name)
            if w is None and g.gt_object_world_ned is not None:
                w = g.gt_object_world_ned[:2]
            if w is not None:
                gt = g.world_to_r(w)
        final_state = est.state()
        stats = {
            "mission_time_s": round(time.monotonic() - t_start, 1),
            "distance_flown_est_m": round(final_state.dist, 1),
            "tier1_counts": perception.counts,
            "tier1_max_p_red": round(perception.max_p_red, 3),
            "tier1_latency": perception.latency_summary(),
            "vlm_calls": verifier.calls,
        }
        true_ne = g.world_to_r(sim.world_ne)
        stats["evaluation_flight"] = {   # evaluation only; ground truth is never used by the mission
            "min_dist_to_target_m": None if min_gt_dist[0] == float("inf") else round(min_gt_dist[0], 1),
            "final_est_minus_true_m": round(math.hypot(final_state.n - true_ne[0], final_state.e - true_ne[1]), 1),
            "waypoint_trace": nav.trace,
        }
        rn, re_ = g.r_world_ne
        gdown = sim.ground_z0
        sim.mark("R", rn, re_, gdown, (0.0, 0.4, 1.0, 1.0))
        if final is not None:
            sim.mark("estimate", rn + final.n, re_ + final.e, gdown, (1.0, 0.0, 0.0, 1.0))
        report = build_report(spec, final, stats, gt)
        path = write_report(report, cfg.results_dir)
        log.info("REPORT: %s  (saved %s)", report["summary"], path)
        sim.disconnect()
        return report
