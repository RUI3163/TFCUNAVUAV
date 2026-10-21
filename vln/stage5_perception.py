"""Stage 5: Tier-1 continuous perception (runs on every frame, budget <= 100 ms).

  5a  YOLO11n (fp16, imgsz 416; YOLOv8n fallback)   -> vehicle boxes
  5b  MobileNetV3-Small on 96x96 crops               -> {red_car, other_car, not_car}
  Event logic: E1 = confident red-car candidate, E2 = uncertain candidate -> both go to Tier-2 (Stage 6).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CLASSES, CameraCfg, Config, PerceptionCfg
from .geometry import pixel_to_ground_offset
from .pose_estimator import DeadReckoner, PoseEstimate
from .sim_client import SimInterface

log = logging.getLogger(__name__)


@dataclass
class Det:
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 (px)
    conf: float


@dataclass
class Candidate:
    frame_idx: int
    t: float
    event: str                               # "E1" | "E2"
    box: Tuple[float, float, float, float]
    det_conf: float
    probs: Dict[str, float]
    est: PoseEstimate                        # pose snapshot taken with THIS frame
    ground_ne: Tuple[float, float]           # R-frame position of box centre (for dedupe)
    context_bgr: np.ndarray                  # padded crop handed to the VLM


# ----------------------------------------------------------------------------- pure logic
def decide_event(probs: np.ndarray, cfg: PerceptionCfg) -> Optional[str]:
    """probs ordered as config.CLASSES = (red_car, other_car, not_car)."""
    p_red = float(probs[0])
    if p_red >= cfg.e1_p_red:
        return "E1"
    if p_red >= cfg.e2_p_red_low or float(np.max(probs)) < cfg.e2_max_p_below:
        return "E2"
    return None


def crop_box(img: np.ndarray, box, pad: float) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * bw)), min(w, int(math.ceil(x2 + pad * bw)))
    y1, y2 = max(0, int(y1 - pad * bh)), min(h, int(math.ceil(y2 + pad * bh)))
    return img[y1:y2, x1:x2]


# ----------------------------------------------------------------------------- models
def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector:
    def __init__(self, cfg: PerceptionCfg):
        from ultralytics import YOLO
        self.cfg, self.dev = cfg, _device()
        try:
            self.model = YOLO(cfg.detector_weights)
            self.name = cfg.detector_weights
        except Exception as exc:
            log.warning("Could not load %s (%s); using fallback %s", cfg.detector_weights, exc, cfg.detector_fallback)
            self.model = YOLO(cfg.detector_fallback)
            self.name = cfg.detector_fallback
        self.half = self.dev.type == "cuda"
        self(np.zeros((225, 400, 3), np.uint8))  # warm-up (cudnn autotune etc.)
        log.info("Detector %s ready (fp16=%s, imgsz=%d)", self.name, self.half, cfg.imgsz)

    def __call__(self, bgr: np.ndarray) -> List[Det]:
        r = self.model.predict(bgr, imgsz=self.cfg.imgsz, conf=self.cfg.det_conf, iou=self.cfg.det_iou,
                               classes=list(self.cfg.det_classes), half=self.half,
                               device=self.dev.index if self.dev.type == "cuda" else "cpu", verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        return [Det(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, conf)]


class CropClassifier:
    """MobileNetV3-Small (ImageNet-pretrained, fine-tuned) -> 3-way softmax in CLASSES order."""

    def __init__(self, cfg: PerceptionCfg):
        import torch
        import torchvision
        self.cfg, self.dev, self.torch = cfg, _device(), torch
        if not os.path.exists(cfg.classifier_weights):
            raise FileNotFoundError(
                f"{cfg.classifier_weights} not found. Train it first: python -m tools.train_classifier --help")
        ck = torch.load(cfg.classifier_weights, map_location="cpu", weights_only=True)
        trained = list(ck["classes"])
        m = torchvision.models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, len(trained))
        m.load_state_dict(ck["state_dict"])
        self.half = self.dev.type == "cuda"
        self.model = m.eval().to(self.dev)
        if self.half:
            self.model.half()
        self.order = [trained.index(c) for c in CLASSES]       # reorder columns to canonical CLASSES
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)
        self(np.zeros((1, 96, 96, 3), np.uint8))
        log.info("Classifier ready (classes in checkpoint: %s)", trained)

    def __call__(self, crops_bgr: List[np.ndarray]) -> np.ndarray:
        torch, s = self.torch, self.cfg.crop_size
        arr = np.stack([cv2.cvtColor(cv2.resize(c, (s, s), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                        for c in crops_bgr])
        x = torch.from_numpy(arr).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std)
        x = x.half() if self.half else x
        with torch.inference_mode():
            p = torch.softmax(self.model(x).float(), dim=1).cpu().numpy()
        return p[:, self.order]


# ----------------------------------------------------------------------------- worker
class PerceptionWorker(threading.Thread):
    """Consumes the latest camera frame (older ones are dropped), updates dead reckoning, detects, classifies."""

    def __init__(self, cfg: Config, sim: SimInterface, est: DeadReckoner, det: Detector,
                 clf: CropClassifier, on_candidate: Callable[[Candidate], bool], stop: threading.Event):
        super().__init__(name="tier1-perception", daemon=True)
        self.cfg, self.sim, self.est, self.det, self.clf = cfg, sim, est, det, clf
        self.on_candidate, self.stop = on_candidate, stop
        self.latencies: List[float] = []
        self.counts = {"frames": 0, "E1": 0, "E2": 0, "submitted": 0, "dets": 0, "frames_with_dets": 0}
        self.timing: Dict[str, List[float]] = {"flow": [], "det": [], "cls": []}
        self.max_p_red = 0.0                                     # best red-car probability seen on ANY detection
        self._dbg_n = 0
        self._last_sub: Optional[Tuple[float, Tuple[float, float]]] = None

    def _rate_ok(self, t: float, ne: Tuple[float, float]) -> bool:
        p = self.cfg.perception
        if self._last_sub is None:
            return True
        lt, lne = self._last_sub
        return (t - lt) >= p.submit_cooldown_s or math.hypot(ne[0] - lne[0], ne[1] - lne[1]) > p.submit_radius_m

    def run(self) -> None:
        p, cam = self.cfg.perception, self.cfg.camera
        last = -1
        while not self.stop.is_set():
            frame = self.sim.wait_frame(last, 0.5)
            if frame is None:
                continue
            last = frame.idx
            t0 = time.perf_counter()
            depth = self.sim.latest_depth() if hasattr(self.sim, "latest_depth") else None
            est = self.est.update_frame(frame.bgr, depth) if depth is not None else self.est.update_frame(frame.bgr)
            t1 = time.perf_counter()
            dets = self.det(frame.bgr)
            t2 = time.perf_counter()
            t_cls = 0.0
            cand: Optional[Candidate] = None
            probs_dbg: Optional[np.ndarray] = None
            if dets:
                self.counts["dets"] += len(dets)
                self.counts["frames_with_dets"] += 1
                crops = [crop_box(frame.bgr, d.box, p.crop_pad) for d in dets]
                keep = [i for i, c in enumerate(crops) if c.size > 0]
                if keep:
                    tc0 = time.perf_counter()
                    probs = self.clf([crops[i] for i in keep])
                    t_cls = time.perf_counter() - tc0
                    probs_dbg = probs
                    self.max_p_red = max(self.max_p_red, float(probs[:, 0].max()))
                    j = int(np.argmax(probs[:, 0]))              # most red-car-like detection
                    event = decide_event(probs[j], p)
                    if event:
                        d = dets[keep[j]]
                        cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                        dn, de = pixel_to_ground_offset(cx, cy, est.alt, est.yaw, cam)
                        ne = (est.n + dn, est.e + de)
                        self.counts[event] += 1
                        cand = Candidate(frame.idx, frame.t, event, d.box, d.conf,
                                         dict(zip(CLASSES, map(float, probs[j]))), est, ne,
                                         crop_box(frame.bgr, d.box, self.cfg.verifier.context_pad))
            lat = (time.perf_counter() - t0) * 1000.0
            self.latencies.append(lat)
            self.timing["flow"].append((t1 - t0) * 1000.0)
            self.timing["det"].append((t2 - t1) * 1000.0)
            self.timing["cls"].append(t_cls * 1000.0)
            self.counts["frames"] += 1
            if lat > p.latency_budget_ms:
                log.warning("Tier-1 latency %.0f ms exceeds %.0f ms budget", lat, p.latency_budget_ms)
            if p.debug_dir and (self.counts["frames"] % max(1, p.debug_every) == 0 or (dets and self._dbg_n % 3 == 0)):
                self._save_debug(frame, dets, probs_dbg)
            if dets:
                self._dbg_n += 1
            if cand and self._rate_ok(frame.t, cand.ground_ne):
                if self.on_candidate(cand):
                    self.counts["submitted"] += 1
                    self._last_sub = (frame.t, cand.ground_ne)

    def _save_debug(self, frame, dets, probs) -> None:
        """Draw detector boxes (+ p_red from the classifier) on the frame and write it to cfg.perception.debug_dir."""
        try:
            os.makedirs(self.cfg.perception.debug_dir, exist_ok=True)
            vis = cv2.resize(frame.bgr, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            for k, d in enumerate(dets):
                x1, y1, x2, y2 = [int(v * 2) for v in d.box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"det {d.conf:.2f}" + (f" red {probs[k, 0]:.2f}" if probs is not None and k < len(probs) else "")
                cv2.putText(vis, txt, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(vis, f"alt {self.est.state().alt:.1f}m N{self.est.state().n:.1f} E{self.est.state().e:.1f}",
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(self.cfg.perception.debug_dir, f"f_{frame.idx:06d}.jpg"), vis)
        except Exception as exc:   # diagnostics must never break the mission
            log.debug("debug frame save failed: %s", exc)

    def latency_summary(self) -> Dict[str, float]:
        if not self.latencies:
            return {}
        a = np.asarray(self.latencies)
        return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
                "frames": int(a.size), "over_budget_frac": float((a > self.cfg.perception.latency_budget_ms).mean()),
                "mean_flow_ms": float(np.mean(self.timing["flow"])), "mean_det_ms": float(np.mean(self.timing["det"])),
                "mean_cls_ms": float(np.mean(self.timing["cls"]))}
