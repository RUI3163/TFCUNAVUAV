"""Optical-flow scale under parallax (trees / roofs closer to the camera than the road)."""
import cv2
import numpy as np
import pytest

from vln.config import CameraCfg, NavCfg
from vln.pose_estimator import DeadReckoner, perspective_to_planar


def _frames():
    """Camera moves 1.0 m forward at 10 m altitude. Ground (10 m) shifts 20 px; a raised surface (6 m) covering
    the left 70 % of the frame shifts 33 px."""
    rng = np.random.default_rng(3)

    def tex(h, w):   # multi-scale texture so LK can lock on at coarse pyramid levels (33 px shifts)
        out = np.zeros((h, w), np.float32)
        for sg in (2, 4, 8, 16):
            out += cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sg) * sg
        return ((out - out.min()) / (out.max() - out.min()) * 255).astype(np.uint8)

    G, P = tex(225 + 80, 400 + 80), tex(225 + 80, 400 + 80)
    split = 280
    f0, f1 = G[40:265, 40:440].copy(), G[20:245, 40:440].copy()          # ground: +20 px down
    f0[:, :split] = P[40:265, 40:40 + split]
    f1[:, :split] = P[7:232, 40:40 + split]                              # raised surface: +33 px down
    depth0 = np.full((225, 400), 10.0, np.float32)
    depth0[:, :split] = 6.0
    to_bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return to_bgr(f0), to_bgr(f1), depth0


def test_altitude_scaling_is_wrong_under_parallax_but_depth_scaling_is_right():
    f0, f1, depth = _frames()
    cam = CameraCfg(depth_is_perspective=False)

    plain = DeadReckoner(cam, NavCfg(use_depth_scale=False), (0.0, 0.0))
    plain.on_pose(10.0, 0.0, 0, 0)
    plain.update_frame(f0, depth)
    n_plain = plain.update_frame(f1, depth).n

    dr = DeadReckoner(cam, NavCfg(use_depth_scale=True), (0.0, 0.0))
    dr.on_pose(10.0, 0.0, 0, 0)
    dr.update_frame(f0, depth)
    n_depth = dr.update_frame(f1, depth).n

    assert n_plain > 1.4                      # altitude scaling: ~1.65 m for a true 1.0 m (your log: 1.5-2x too far)
    assert n_depth == pytest.approx(1.0, abs=0.15)


def test_perspective_to_planar_flat_ground():
    cam = CameraCfg()
    h = 10.0
    u = np.arange(cam.width, dtype=np.float32) - cam.cx
    v = np.arange(cam.height, dtype=np.float32) - cam.cy
    persp = h * np.sqrt(1.0 + (u[None, :] ** 2 + v[:, None] ** 2) / cam.fx ** 2)
    assert persp[0, 0] / persp[112, 200] == pytest.approx(1.52, abs=0.02)   # corner/centre ratio
    assert np.allclose(perspective_to_planar(persp, cam), h, atol=1e-3)
