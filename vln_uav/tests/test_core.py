"""Simulator-free, GPU-free unit tests:  pytest -q"""
import math

import numpy as np
import pytest

from vln.config import CameraCfg, Config, GeometryCfg, NavCfg, PerceptionCfg, VerifierCfg, LocalizationCfg
from vln.geometry import pixel_to_ground_offset, range_bearing
from vln.pose_estimator import DeadReckoner
from vln.stage3_planner import Rect, plan_search, footprint, OccupancyGrid, astar, PlanningError, coverage_margin
from vln.stage5_perception import decide_event, Candidate
from vln.stage6_verifier import VerifyResult
from vln.stage7_localizer import ConsensusGate, Localizer
from vln.stage2_mission import MissionSpec


def test_camera_intrinsics():
    cam = CameraCfg()
    assert cam.fx == pytest.approx(200.0)          # 400 px, 90 deg HFOV
    w, h = footprint(10.0, cam)
    assert (w, h) == (pytest.approx(20.0), pytest.approx(11.25))


def test_pinhole_axes():
    cam, alt = CameraCfg(), 10.0
    dn, de = pixel_to_ground_offset(cam.cx, cam.cy, alt, 0.0, cam)       # image centre = nadir
    assert (abs(dn), abs(de)) == (pytest.approx(0), pytest.approx(0))
    dn, de = pixel_to_ground_offset(cam.cx, 0.0, alt, 0.0, cam)           # image top -> forward (north)
    assert (dn, de) == (pytest.approx(112.5 / 200 * 10), pytest.approx(0, abs=1e-9))  # 5.625 m ahead
    dn, de = pixel_to_ground_offset(cam.width, cam.cy, alt, 0.0, cam)    # right edge -> +10 m east
    assert (dn, de) == (pytest.approx(0, abs=1e-9), pytest.approx(10.0))
    dn, de = pixel_to_ground_offset(cam.width, cam.cy, alt, math.pi / 2, cam)  # yaw east: image-right = south
    assert (dn, de) == (pytest.approx(-10.0), pytest.approx(0, abs=1e-9))


def test_range_bearing():
    assert range_bearing(10, 0) == (pytest.approx(10), pytest.approx(0))
    r, b = range_bearing(0, -5)
    assert (r, b) == (pytest.approx(5), pytest.approx(270))


def test_lawnmower_covers_area():
    g, cam = GeometryCfg(area_center_ne=(0.0, 0.0), start_offset_ne=(0.0, 0.0)), CameraCfg()   # generic R-centred square
    wps = plan_search(g, cam)
    easts = sorted({round(e, 1) for _, e in wps})
    assert easts == [-10.0, 0.0, 10.0]                                 # 3 lanes for 40 m @ 20 m swath
    assert all(abs(n) <= 20 + 1e-6 and abs(e) <= 20 + 1e-6 for n, e in wps)


def test_astar_detours_around_no_fly_box():
    g = GeometryCfg(area_center_ne=(0.0, 0.0), no_fly_boxes=[(-5.0, -3.0, 5.0, 3.0)], start_offset_ne=(-20.0, -10.0))
    wps = plan_search(g, CameraCfg())
    grid = OccupancyGrid(Rect(-24, -24, 24, 24), 1.0, g.no_fly_boxes, g.safety_margin_m)
    pts = [g.start_offset_ne] + wps
    assert all(grid.segment_free(a, b) for a, b in zip(pts, pts[1:]))
    assert len(wps) > 6                                                # extra detour waypoints were inserted


def test_astar_rejects_blocked_goal():
    grid = OccupancyGrid(Rect(-10, -10, 10, 10), 1.0, [(-2, -2, 2, 2)], 0.0)
    with pytest.raises(PlanningError):
        astar(grid, (-9, -9), (0, 0))


def test_decide_event():
    p = PerceptionCfg()
    assert decide_event(np.array([0.9, 0.05, 0.05]), p) == "E1"
    assert decide_event(np.array([0.4, 0.4, 0.2]), p) == "E2"          # p_red in the uncertainty band
    assert decide_event(np.array([0.05, 0.9, 0.05]), p) is None
    assert decide_event(np.array([0.1, 0.4, 0.4]), p) == "E2"          # unsure overall


def test_dead_reckoning_recovers_known_translation():
    """Shift a textured image 20 px 'down' (camera moved forward) -> north displacement = 20*alt/fy."""
    import cv2
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur((rng.random((225 + 40, 400 + 40)) * 255).astype(np.uint8), (5, 5), 0)
    f0 = cv2.cvtColor(base[20:245, 20:420], cv2.COLOR_GRAY2BGR)
    f1 = cv2.cvtColor(base[0:225, 20:420], cv2.COLOR_GRAY2BGR)         # content moved down by 20 px
    dr = DeadReckoner(CameraCfg(), NavCfg(), (0.0, 0.0))
    dr.on_pose(10.0, 0.0, 0, 0)
    dr.update_frame(f0)
    s = dr.update_frame(f1)
    assert s.flow_ok
    assert s.n == pytest.approx(20 * 10.0 / 200.0, abs=0.15)           # +1.0 m north
    assert s.e == pytest.approx(0.0, abs=0.15)


def _res(n_e, is_target, idx):
    from vln.pose_estimator import PoseEstimate
    est = PoseEstimate(0.0, 0.0, 10.0, 0.0, 30.0, True)
    cam = CameraCfg()
    # place box so ground offset == n_e (north, east) at yaw 0
    n, e = n_e
    v, u = cam.cy - n / 10.0 * cam.fy, cam.cx + e / 10.0 * cam.fx
    cand = Candidate(idx, 0.0, "E1", (u - 5, v - 5, u + 5, v + 5), 0.9, {}, est, (n, e), np.zeros((4, 4, 3), np.uint8))
    return VerifyResult(cand, 0.9 if is_target else 0.1, is_target, 100.0)


def test_consensus_gate_n3():
    gate = ConsensusGate(Localizer(CameraCfg(), LocalizationCfg()), VerifierCfg())
    assert gate.add(_res((5.0, 5.0), True, 1)) is None
    assert gate.add(_res((5.2, 4.9), True, 2)) is None
    fin = gate.add(_res((4.9, 5.1), True, 3))
    assert fin is not None and fin.n_frames == 3
    assert fin.n == pytest.approx(5.03, abs=0.1) and fin.e == pytest.approx(5.0, abs=0.1)
    assert fin.sigma_m > 0.8                                           # dominated by 3% drift of 30 m = 0.9 m


def test_consensus_resets_on_negative_and_disagreement():
    gate = ConsensusGate(Localizer(CameraCfg(), LocalizationCfg()), VerifierCfg())
    gate.add(_res((5, 5), True, 1)); gate.add(_res((5, 5), True, 2))
    gate.add(_res((5, 5), False, 3))                                    # negative -> reset
    assert gate.add(_res((5, 5), True, 4)) is None
    gate.add(_res((5, 5), True, 5))
    assert gate.add(_res((-15, 12), True, 6)) is None                   # far away -> restart streak


def test_mission_spec_validation():
    ok = {"task": "search_and_report", "target": {"category": "passenger_car", "color": "red"},
          "search_region": {"area_type": "parking_area", "near_reference": "R"}, "reference": {"id": "R"},
          "report_fields": ["east_m", "north_m"]}
    assert MissionSpec.model_validate(ok).target.color == "red"
    bad = dict(ok, reference={"id": "Q"})
    with pytest.raises(Exception):
        MissionSpec.model_validate(bad)


def test_start_outside_area_gets_straight_transit():
    g = GeometryCfg(area_center_ne=(0.0, 0.0), start_offset_ne=(150.0, 200.0))
    wps = plan_search(g, CameraCfg())
    assert len(wps) == 6 and abs(wps[0][0]) <= 20 and abs(wps[0][1]) <= 20


# ---- Car_65 world coordinates / Reference Point R -------------------------------------------------
def test_car65_maps_into_search_area():
    g = GeometryCfg()
    assert g.r_world_ne == pytest.approx((-130.682, -174.213))
    n, e = g.gt_object_in_r()
    assert (n, e) == (pytest.approx(12.0, abs=1e-6), pytest.approx(26.0, abs=1e-6))
    assert g.area_bounds() == (0.0, 0.0, 40.0, 40.0)                  # R = south-west corner of the square
    assert g.in_area((n, e), margin=10.0)                              # comfortably inside, not on the edge
    assert g.ground_z_world == pytest.approx(-1.034)
    assert g.r_to_world((12.0, 26.0)) == pytest.approx((-118.682, -148.213))


def test_square_centred_on_r_would_miss_car65():
    g = GeometryCfg(area_center_ne=(0.0, 0.0))                        # the old default: east reach is only +-20 m
    assert not g.in_area(g.gt_object_in_r())


def test_default_sweep_covers_car65_with_margin():
    g, cam = GeometryCfg(), CameraCfg()
    wps = plan_search(g, cam)
    assert all(-1e-6 <= n <= 40 + 1e-6 and -1e-6 <= e <= 40 + 1e-6 for n, e in wps)
    assert coverage_margin(wps, g.start_offset_ne, g.gt_object_in_r(), g.altitude_m, cam) > 5.0
    assert coverage_margin(wps, g.start_offset_ne, (12.0, 80.0), g.altitude_m, cam) < 0     # far east: not swept


def test_start_offset_follows_spawn_pose():
    g = GeometryCfg()
    assert g.start_offset_ne == pytest.approx((0.0, 0.0))             # default spawn is over R
    assert g.apply_spawn((0.0, 0.0)) == pytest.approx((130.682, 174.213))     # old spawn at world origin: far transit
    g.start_offset_ne = (0.0, 0.0)
    assert g.apply_spawn((-130.682, -174.213)) == pytest.approx((0.0, 0.0))   # spawn exactly over R
    wps = plan_search(g, CameraCfg())                                  # start (0,0) is inside the area now
    assert coverage_margin(wps, g.start_offset_ne, (12.0, 26.0), g.altitude_m, CameraCfg()) > 5.0


def test_perception_worker_counts_detections_and_saves_debug(tmp_path):
    """Tier-1 diagnostics: raw detections are counted even when no E1/E2 event fires; debug frames get written."""
    import threading
    from vln.config import Config
    from vln.pose_estimator import PoseEstimate
    from vln.sim_client import Frame
    from vln.stage5_perception import Det, PerceptionWorker

    cfg = Config()
    cfg.perception.debug_dir = str(tmp_path)
    cfg.perception.debug_every = 2
    stop = threading.Event()

    class FakeSim:
        def __init__(self): self.i = 0
        def wait_frame(self, last, timeout):
            self.i += 1
            if self.i > 6:
                stop.set()
                return None
            return Frame(self.i, float(self.i), np.zeros((225, 400, 3), np.uint8))

    class FakeEst:
        def update_frame(self, bgr): return self.state()
        def state(self): return PoseEstimate(0.0, 0.0, 10.0, 0.0, 0.0, True)

    det = lambda bgr: [Det((100.0, 80.0, 160.0, 120.0), 0.4)]
    clf = lambda crops: np.tile(np.array([0.05, 0.9, 0.05]), (len(crops), 1))      # confident 'other_car'
    w = PerceptionWorker(cfg, FakeSim(), FakeEst(), det, clf, lambda c: True, stop)
    w.run()
    assert w.counts["frames"] == 6 and w.counts["dets"] == 6 and w.counts["frames_with_dets"] == 6
    assert w.counts["E1"] == 0 and w.counts["E2"] == 0 and w.counts["submitted"] == 0
    assert w.max_p_red == pytest.approx(0.05)
    assert len(list(tmp_path.glob("*.jpg"))) >= 3


def test_navigator_records_est_vs_true_trace():
    import asyncio, threading
    from vln.config import Config
    from vln.pose_estimator import PoseEstimate
    from vln.stage4_navigation import Navigator

    cfg = Config()
    cfg.nav.nav_mode = "estimate"

    class FakeSim:
        ground_z0 = -1.034
        def __init__(self): self.world_ne = cfg.geometry.r_to_world((0.0, 0.0)); self.alt = 10.0
        async def velocity_z(self, vn, ve, z, dur):
            self.world_ne = (self.world_ne[0] + vn * dur, self.world_ne[1] + ve * dur)
        async def hover(self): pass

    sim = FakeSim()

    class FakeEst:                       # estimate has a constant 2 m north bias vs the truth
        lost_frames = 0
        def state(self):
            n, e = cfg.geometry.world_to_r(sim.world_ne)
            return PoseEstimate(n + 2.0, e, 10.0, 0.0, 0.0, True)

    nav = Navigator(cfg, sim, FakeEst())
    assert asyncio.run(nav.fly_path([(0.0, 10.0), (0.0, 20.0)], threading.Event()))
    assert len(nav.trace) == 2 and all(t["est_minus_true_m"] == pytest.approx(2.0, abs=0.05) for t in nav.trace)
