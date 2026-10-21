"""Fly a lawn-mower sweep (ground-truth position feedback) and save nadir RGB frames for classifier / detector training.

  python -m tools.collect_frames --out data/raw --altitudes 8 10 12 --every 5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading

import cv2

from vln.config import Config
from vln.pose_estimator import DeadReckoner
from vln.sim_client import SimInterface
from vln.stage3_planner import plan_search
from vln.stage4_navigation import Navigator


async def main_async(a) -> None:
    cfg = Config()
    cfg.nav.nav_mode = "groundtruth"
    os.makedirs(a.out, exist_ok=True)
    sim = SimInterface(cfg)
    sim.connect()
    if cfg.geometry.auto_start_from_sim:
        cfg.geometry.apply_spawn(sim.start_world_ne)   # start offset relative to R from the live spawn pose
    est = DeadReckoner(cfg.camera, cfg.nav, cfg.geometry.start_offset_ne)
    sim.add_pose_listener(est.on_pose)
    sim.add_imu_listener(est.on_imu)
    sim.start_streams()
    stop = threading.Event()
    saved = {"n": 0}

    def saver() -> None:
        last, k = -1, 0
        while not stop.is_set():
            f = sim.wait_frame(last, 0.5)
            if f is None:
                continue
            last, k = f.idx, k + 1
            if k % a.every == 0:
                cv2.imwrite(os.path.join(a.out, f"f_{f.idx:06d}.png"), f.bgr)
                saved["n"] += 1

    th = threading.Thread(target=saver, daemon=True)
    th.start()
    try:
        for alt in a.altitudes:
            cfg.geometry.altitude_m = alt
            nav = Navigator(cfg, sim, est)
            if alt == a.altitudes[0]:
                await nav.takeoff_and_climb()
            else:  # change altitude in place: climb/descend to the new level, then sweep again
                await nav.climb_to(alt)
            await nav.fly_path(plan_search(cfg.geometry, cfg.camera), threading.Event())
    finally:
        stop.set()
        await sim.land()
        sim.disconnect()
        print(f"saved {saved['n']} frames to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--altitudes", type=float, nargs="+", default=[10.0])
    ap.add_argument("--every", type=int, default=5, help="save every k-th frame")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_async(ap.parse_args()))
