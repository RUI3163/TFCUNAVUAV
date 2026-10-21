"""Thin wrapper around the Project AirSim Python client (`projectairsim`).

Mirrors the connection pattern of the reference script (ProjectAirSimClient ->
World(scene) -> Drone(name) -> enable_api_control/arm -> *_async + await task)
and adds latest-frame image streaming + pose/IMU subscriptions.

Threading model: Project AirSim invokes topic callbacks on its own receive thread;
callbacks here only copy data and return (as in the official examples).
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .config import Config
from .geometry import quat_to_yaw

log = logging.getLogger(__name__)


KNOWN_SIM_CONFIG = r"D:\AirSim\ProjectAirSim\client\python\example_user_scripts\sim_config"


def resolve_sim_config_path(user_path: Optional[str], scene_file: str) -> Optional[str]:
    """Folder that directly contains scene_file (what World(sim_config_path=...) needs, as used by
    find_object_coords.py / fly_above_object.py). Tries the given path, its ./sim_config child and parent."""
    bases = [b for b in (user_path, "sim_config", ".", KNOWN_SIM_CONFIG) if b]
    for b in bases:
        for c in (b, os.path.join(b, "sim_config"), os.path.dirname(os.path.normpath(b))):
            if os.path.isfile(os.path.join(c, scene_file)):
                return c
    return None


class SimConnectionError(RuntimeError):
    pass


@dataclass
class Frame:
    idx: int
    t: float            # time.monotonic() at receipt
    bgr: np.ndarray     # HxWx3 uint8, OpenCV (BGR) order


class SimInterface:
    LAND_HANDOFF_M = 3.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = None
        self.world = None
        self.drone = None
        self.ground_z0 = 0.0                      # world NED z of the vehicle while on the ground
        self.start_world_ne = (0.0, 0.0)
        self._cond = threading.Condition()
        self._frame: Optional[Frame] = None
        self._frame_idx = 0
        self._pose_listeners: list[Callable[[float, float, float, float], None]] = []
        self._imu_listeners: list[Callable[[float, float], None]] = []
        self._pose_seen = threading.Event()
        self._stop = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._warned_size = False
        self.altitude_m = 0.0
        self.world_ne = (0.0, 0.0)               # last sim pose (world NED); used ONLY by groundtruth nav mode

    # ------------------------------------------------------------------ connect
    def connect(self) -> None:
        try:
            from projectairsim import Drone, ProjectAirSimClient, World
        except ImportError as exc:  # pragma: no cover
            raise SimConnectionError(
                "projectairsim not installed. pip install -e ProjectAirSim/client/python/projectairsim"
            ) from exc
        s = self.cfg.sim
        try:
            self.client = ProjectAirSimClient()
            self.client.connect()
            cfg_dir = resolve_sim_config_path(s.sim_config_path, s.scene_config)
            if cfg_dir is None:
                raise SimConnectionError(f"Could not find {s.scene_config}; pass --sim-config-path "
                                         "(the folder that directly contains the scene .jsonc).")
            log.info("Using sim config folder: %s", cfg_dir)
            self.world = World(self.client, s.scene_config, delay_after_load_sec=s.delay_after_load_sec,
                               sim_config_path=cfg_dir)
            self.drone = Drone(self.client, self.world, s.vehicle_name)
        except Exception as exc:
            raise SimConnectionError(f"Could not connect to Project AirSim: {exc}") from exc

        # Subscriptions (callbacks only copy + return).
        self.client.subscribe(self.drone.robot_info["actual_pose"], self._on_pose)
        self.client.subscribe(self.drone.sensors[s.camera_id]["scene_camera"], self._on_image)
        try:
            self.client.subscribe(self.drone.sensors[s.imu_id]["imu_kinematics"], self._on_imu)
        except KeyError:
            log.warning("IMU '%s' not found in robot config; yaw_source must be 'pose'.", s.imu_id)

        if not self._pose_seen.wait(5.0):
            raise SimConnectionError("No actual_pose messages received within 5 s.")
        log.info("Connected. ground_z0=%.2f start_world_ne=%s", self.ground_z0, self.start_world_ne)

    def start_streams(self) -> None:
        """Verify image streaming works; otherwise fall back to polling get_images()."""
        if self.wait_frame(-1, self.cfg.sim.frame_wait_timeout_s) is None:
            log.warning("No streamed camera frames; falling back to polling get_images().")
            self._poll_thread = threading.Thread(target=self._poll_images, daemon=True)
            self._poll_thread.start()
            if self.wait_frame(-1, 5.0) is None:
                raise SimConnectionError("Camera produced no frames (check capture-enabled in robot config).")

    def disconnect(self) -> None:
        self._stop.set()
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:  # pragma: no cover
                pass

    # ---------------------------------------------------------------- callbacks
    def _on_pose(self, _topic, msg) -> None:
        p, q = msg["position"], msg["orientation"]
        if not self._pose_seen.is_set():
            gz = self.cfg.geometry.ground_z_world
            self.ground_z0 = float(p["z"]) if gz is None else float(gz)
            self.start_world_ne = (float(p["x"]), float(p["y"]))
            self._pose_seen.set()
        yaw = quat_to_yaw(q["w"], q["x"], q["y"], q["z"])
        self.world_ne = (float(p["x"]), float(p["y"]))
        altitude = self.ground_z0 - float(p["z"])      # NED: up is negative z
        self.altitude_m = altitude
        for fn in self._pose_listeners:
            fn(altitude, yaw, float(p["x"]), float(p["y"]))

    def _on_imu(self, _topic, msg) -> None:
        wz = float(msg["angular_velocity"]["z"])
        t = time.monotonic()
        for fn in self._imu_listeners:
            fn(wz, t)

    def _on_image(self, _topic, msg) -> None:
        self._publish_frame(msg)

    def _publish_frame(self, msg) -> None:
        from projectairsim.utils import unpack_image
        img = unpack_image(msg)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = np.repeat(img.reshape(img.shape[0], img.shape[1], 1), 3, axis=2)
        cam = self.cfg.camera
        if cam.swap_rgb:
            img = img[:, :, ::-1]
        if not self._warned_size and (img.shape[1], img.shape[0]) != (cam.width, cam.height):
            log.warning("Camera frame is %dx%d but CameraCfg says %dx%d -- fix config or robot jsonc.",
                        img.shape[1], img.shape[0], cam.width, cam.height)
            self._warned_size = True
        with self._cond:
            self._frame_idx += 1
            self._frame = Frame(self._frame_idx, time.monotonic(), np.ascontiguousarray(img))
            self._cond.notify_all()

    def _poll_images(self) -> None:
        from projectairsim.types import ImageType
        while not self._stop.is_set():
            try:
                imgs = self.drone.get_images(self.cfg.sim.camera_id, [ImageType.SCENE])
                self._publish_frame(next(iter(imgs.values())))
            except Exception as exc:  # pragma: no cover
                log.debug("poll get_images failed: %s", exc)
            time.sleep(0.05)

    # -------------------------------------------------------------- frame access
    def wait_frame(self, last_idx: int, timeout: float) -> Optional[Frame]:
        """Block until a frame newer than last_idx exists (latest-only: stale frames are dropped)."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._frame is not None and self._frame.idx > last_idx, timeout)
            return self._frame if ok else None

    def add_pose_listener(self, fn) -> None:
        self._pose_listeners.append(fn)

    def add_imu_listener(self, fn) -> None:
        self._imu_listeners.append(fn)

    # ------------------------------------------------------------------ control
    async def takeoff(self) -> None:
        self.drone.enable_api_control()
        self.drone.arm()
        await (await self.drone.takeoff_async())

    async def velocity_z(self, v_north: float, v_east: float, z_world: float, duration: float) -> None:
        """Horizontal velocity command + P-controlled vertical rate toward z_world (NED).
        Same pattern as fly_above_object.go_to (move_by_velocity_async, vd = clip(0.7*err, +-3)), which is
        proven on your setup. Yaw rate defaults to 0 => heading held."""
        z_now = self.ground_z0 - self.altitude_m
        v_down = max(-3.0, min(3.0, 0.7 * (z_world - z_now)))
        await (await self.drone.move_by_velocity_async(v_north, v_east, v_down, duration=duration))

    async def hover(self) -> None:
        await (await self.drone.hover_async())

    async def land(self) -> None:
        """Staged landing. land_async() alone descends at ~0.19 m/s (your driver measured ~97 s from
        15-18 m), so descend quickly with velocity commands to the handoff height first."""
        t0 = time.monotonic()
        while self.altitude_m > self.LAND_HANDOFF_M + 0.5 and time.monotonic() - t0 < 30:
            await (await self.drone.move_by_velocity_async(0.0, 0.0, 3.0, duration=0.5))
        await (await self.drone.land_async())
        self.drone.disarm()
        self.drone.disable_api_control()

    # --------------------------------------------------------- evaluation helper
    def get_object_ne(self, name: str) -> Optional[tuple[float, float]]:
        """Ground-truth object position (world NED). For offline EVALUATION only."""
        try:
            pose = self.world.get_object_pose(name)
            t = pose["translation"]
            return float(t["x"]), float(t["y"])
        except Exception as exc:
            log.warning("Could not read ground-truth pose of '%s': %s", name, exc)
            return None

    def mark(self, label: str, north: float, east: float, down: float, rgba=(1.0, 0.0, 0.0, 1.0),
             duration: float = 900.0) -> None:
        """Draw a labelled debug point in the Unreal world (visual check of R / estimated / true target)."""
        try:
            self.world.plot_debug_strings([label], [[north, east, down - 2.0]], scale=1.5,
                                          color_rgba=list(rgba), duration=duration)
            self.world.plot_debug_points([[north, east, down - 2.0]], color_rgba=list(rgba), size=15.0,
                                         duration=duration, is_persistent=False)
        except Exception as exc:
            log.warning("Could not draw debug marker '%s': %s", label, exc)
