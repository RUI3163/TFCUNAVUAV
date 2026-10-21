"""Central configuration. Every tunable lives here; nothing is hard-coded elsewhere.

Frame conventions (used consistently across all modules)
  * NED: north (+x), east (+y), down (+z); altitude = -z.
  * "R frame": origin at Reference Point R, axes parallel to world north/east.
  * yaw: clockwise from north, radians (NED convention).
  * Nadir camera, yaw-locked mission: image-up == body-forward, image-right == body-right.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MISSION_PROMPT = (
    "Search the parking area near Reference Point R for a red passenger car "
    "and report its location relative to Reference Point R."
)

CLASSES = ("red_car", "other_car", "not_car")  # canonical names; checkpoint stores its own index map

# --------------------------------------------------------------------------------------------------
# World <-> R-frame geometry for this scene (Project AirSim world NED, metres).
#
# MEASURED (find_object_coords.py): Car_65 (ground-truth, evaluation only)
CAR65_WORLD_NED: Tuple[float, float, float] = (-118.682, -148.213, -1.034)   # north, east, down
CAR65_YAW_DEG: float = 87.55                                                  # informational only
# DESIGN CHOICE: where Car_65 should land in the R frame (R = local origin [0,0]).
CAR65_IN_R_NE: Tuple[float, float] = (12.0, 26.0)                             # north, east of R
# => Reference Point R in world NED:  R = Car_65 - (12, 26)
R_WORLD_NE: Tuple[float, float] = (round(CAR65_WORLD_NED[0] - CAR65_IN_R_NE[0], 3),   # -130.682
                                   round(CAR65_WORLD_NED[1] - CAR65_IN_R_NE[1], 3))   # -174.213
# 40 x 40 m search square, OFFSET from R (R is its south-west corner): north 0..40, east 0..40.
# Car_65 at (12, 26) is inside it, 12 m from the south edge and 14 m from the east edge.
SEARCH_AREA_CENTER_R: Tuple[float, float] = (20.0, 20.0)
# Where the scene spawns the drone in world NED: directly over Reference Point R (the south-west corner of the
# search square), so the start is INSIDE the area -> no unchecked transit leg, no transit drift. This must match
# the Drone1 "origin" x/y in your scene jsonc; at run time the real spawn pose read from the simulator overrides
# it (GeometryCfg.auto_start_from_sim).
SPAWN_WORLD_NE_ASSUMED: Tuple[float, float] = R_WORLD_NE
START_OFFSET_R: Tuple[float, float] = (round(SPAWN_WORLD_NE_ASSUMED[0] - R_WORLD_NE[0], 3),   # 0.0
                                       round(SPAWN_WORLD_NE_ASSUMED[1] - R_WORLD_NE[1], 3))   # 0.0
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraCfg:
    width: int = 400
    height: int = 225
    hfov_deg: float = 90.0
    swap_rgb: bool = False       # set True if saved frames show wrong colours (red car looks blue)

    @property
    def fx(self) -> float:  # focal length in pixels (square pixels => fy == fx)
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0


@dataclass
class SimCfg:
    scene_config: str = "scene_basic_drone.jsonc"   # <-- your scene (must contain Car_65)
    sim_config_path: Optional[str] = None           # folder DIRECTLY containing the scene jsonc (auto-resolved if None)
    vehicle_name: str = "Drone1"
    camera_id: str = "DownCamera"
    imu_id: str = "IMU1"
    delay_after_load_sec: int = 2
    frame_wait_timeout_s: float = 3.0               # then fall back to polling get_images()


@dataclass
class GeometryCfg:
    area_size_m: float = 40.0                       # 40 x 40 m square
    # Centre of the search square in the R frame. NOT (0,0): the square is offset from R so that it
    # covers Car_65 at (N 12, E 26) -- a square centred on R would only reach east +-20 m.
    area_center_ne: Tuple[float, float] = SEARCH_AREA_CENTER_R
    altitude_m: float = 10.0
    cruise_speed_mps: float = 2.5
    lane_overlap: float = 0.2                       # fraction of swath overlapped between lanes
    waypoint_tol_m: float = 1.0
    # Reference Point R in the sim's world NED frame (north, east). R = Car_65 - (12, 26).
    r_world_ne: Tuple[float, float] = R_WORLD_NE
    # UAV start relative to R (R frame) = spawn_world - r_world. Static fallback assumes the drone spawns at
    # world (0, 0); with auto_start_from_sim=True it is overwritten from the real spawn pose after connect.
    start_offset_ne: Tuple[float, float] = START_OFFSET_R
    auto_start_from_sim: bool = True
    # Optional no-fly boxes (n_min, e_min, n_max, e_max) in the R frame -> A* detours.
    no_fly_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    safety_margin_m: float = 2.0
    grid_res_m: float = 1.0
    # World NED z of the GROUND AT THE SEARCH AREA (not at the spawn point). Altitude AGL = ground_z - z.
    # Car_65's base was measured at down = -1.034 m; use that (or the local road level).
    ground_z_world: Optional[float] = CAR65_WORLD_NED[2]
    # Evaluation only; never used by the search/perception/localisation pipeline.
    gt_object_name: Optional[str] = "Car_65"
    gt_object_world_ned: Optional[Tuple[float, float, float]] = CAR65_WORLD_NED   # fallback if the sim lookup fails
    gt_object_yaw_deg: Optional[float] = CAR65_YAW_DEG                            # informational

    # ---- frame helpers (world NED <-> R frame; axes are parallel, only the origin differs) ----
    def world_to_r(self, world_ne: Tuple[float, float]) -> Tuple[float, float]:
        return world_ne[0] - self.r_world_ne[0], world_ne[1] - self.r_world_ne[1]

    def r_to_world(self, r_ne: Tuple[float, float]) -> Tuple[float, float]:
        return r_ne[0] + self.r_world_ne[0], r_ne[1] + self.r_world_ne[1]

    def area_bounds(self) -> Tuple[float, float, float, float]:
        """(n_min, e_min, n_max, e_max) of the search square in the R frame."""
        h, (cn, ce) = self.area_size_m / 2.0, self.area_center_ne
        return cn - h, ce - h, cn + h, ce + h

    def in_area(self, r_ne: Tuple[float, float], margin: float = 0.0) -> bool:
        n0, e0, n1, e1 = self.area_bounds()
        return n0 + margin <= r_ne[0] <= n1 - margin and e0 + margin <= r_ne[1] <= e1 - margin

    def gt_object_in_r(self) -> Optional[Tuple[float, float]]:
        """Configured ground-truth object position in the R frame (static, no simulator needed)."""
        if self.gt_object_world_ned is None:
            return None
        return self.world_to_r((self.gt_object_world_ned[0], self.gt_object_world_ned[1]))

    def apply_spawn(self, spawn_world_ne: Tuple[float, float]) -> Tuple[float, float]:
        """Set start_offset_ne from the drone's real spawn pose (world NED). Returns the new offset."""
        self.start_offset_ne = self.world_to_r(spawn_world_ne)
        return self.start_offset_ne


@dataclass
class NavCfg:
    nav_mode: str = "estimate"      # "estimate": closed loop on dead-reckoning | "groundtruth": same controller on sim pose (debug/data collection)
    control_dt_s: float = 0.2
    kp: float = 0.8
    yaw_source: str = "gyro"        # "gyro" (integrated IMU) | "pose" (sim attitude, AHRS stand-in)
    gyro_yaw_sign: float = 1.0      # flip to -1 if your IMU frame integrates the wrong way
    flow_max_corners: int = 150
    flow_min_features: int = 8
    climb_tol_m: float = 0.5


@dataclass
class PerceptionCfg:
    detector_weights: str = "yolo11n.pt"
    detector_fallback: str = "yolov8n.pt"
    imgsz: int = 416
    det_conf: float = 0.15
    det_iou: float = 0.5
    det_classes: Tuple[int, ...] = (2,)             # COCO 'car'. Use your own ids if you fine-tune.
    classifier_weights: str = "weights/mobilenetv3_small_cls.pt"
    crop_size: int = 96
    crop_pad: float = 0.15
    latency_budget_ms: float = 100.0
    e1_p_red: float = 0.60                          # E1: confident red-car candidate
    e2_p_red_low: float = 0.25                      # E2: 0.25 <= p_red < 0.60 ...
    e2_max_p_below: float = 0.50                    # ... or the classifier is unsure overall
    submit_cooldown_s: float = 1.0                  # min time between VLM submissions near same spot
    submit_radius_m: float = 2.0
    # Diagnostics: if set, Tier-1 saves every `debug_every`-th frame (and every frame with detections, rate-limited)
    # with the detector boxes and classifier probabilities drawn on it. Off by default.
    debug_dir: Optional[str] = None
    debug_every: int = 15


@dataclass
class VerifierCfg:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    load_4bit: bool = True                          # fp16 weights (~4.4 GB) do not fit in 4 GB VRAM
    queue_size: int = 3
    context_pad: float = 0.5                        # extra context around the bbox for the VLM crop
    min_side_px: int = 224
    p_yes_threshold: float = 0.5
    consensus_n: int = 3                            # N consecutive agreeing verified frames
    agree_radius_m: float = 3.0
    prompt: str = (
        "This is a top-down aerial view of a parking area. Is the vehicle in the "
        "centre of this image a {color} passenger car (not a truck, bus, or other colour)? "
        "Answer with exactly one word: Yes or No."
    )


@dataclass
class MissionLLMCfg:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    load_4bit: bool = True
    max_new_tokens: int = 256
    max_retries: int = 3


@dataclass
class LocalizationCfg:
    pixel_sigma_px: float = 3.0
    altitude_sigma_m: float = 0.3
    drift_frac: float = 0.03                        # dead-reckoning drift as fraction of distance flown
    yaw_sigma_rad: float = math.radians(2.0)


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    geometry: GeometryCfg = field(default_factory=GeometryCfg)
    nav: NavCfg = field(default_factory=NavCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    verifier: VerifierCfg = field(default_factory=VerifierCfg)
    mission_llm: MissionLLMCfg = field(default_factory=MissionLLMCfg)
    localization: LocalizationCfg = field(default_factory=LocalizationCfg)
    prompt: str = MISSION_PROMPT
    results_dir: str = "results"
    logs_dir: str = "logs"
