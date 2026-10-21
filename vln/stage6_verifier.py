"""Stage 6: Tier-2 VLM verifier (Qwen2-VL-2B-Instruct, no fallback), asynchronous + event driven.

Only invoked on E1/E2 events. Runs in its own thread behind a bounded queue so Tier-1 and flight
never block on it. Verdict = P(Yes) from ONE forward pass (next-token logits of "Yes" vs "No"),
which is faster and better calibrated than parsing generated text.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .config import VerifierCfg
from .gpu import bnb_4bit_config, log_vram
from .stage5_perception import Candidate

log = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    cand: Candidate
    p_yes: float
    is_target: bool
    latency_ms: float


class VLMVerifier:
    def __init__(self, cfg: VerifierCfg, target_color: str):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.cfg, self.torch = cfg, torch
        self.prompt = cfg.prompt.format(color=target_color.upper())
        kw = dict(quantization_config=bnb_4bit_config()) if cfg.load_4bit else dict(torch_dtype=torch.float16)
        log.info("Stage 6: loading %s (4-bit=%s)", cfg.model_id, cfg.load_4bit)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, device_map={"": 0}, **kw).eval()
        # Small pixel range keeps visual tokens (and VRAM / latency) low.
        self.proc = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=224 * 224, max_pixels=336 * 336)
        tk = self.proc.tokenizer
        self.yes_id, self.no_id = tk.encode("Yes")[0], tk.encode("No")[0]
        self.target_color = target_color.capitalize()
        self.color_ids = []
        if cfg.mode == "color":
            names = [c for c in cfg.colors]
            ids = [tk.encode(c)[0] for c in names]
            if len(set(ids)) != len(ids):
                dup = [n for n, i in zip(names, ids) if ids.count(i) > 1]
                log.warning("Colour words share a first token (%s); their probabilities are merged.", dup)
            if self.target_color not in names:
                raise ValueError(f"target colour {self.target_color!r} not in VerifierCfg.colors")
            self.color_names, self.color_ids = names, ids
        question = cfg.color_prompt if cfg.mode == "color" else self.prompt
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        self.text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        log_vram("stage6 loaded")

    def p_yes(self, bgr: np.ndarray) -> float:
        from PIL import Image
        h, w = bgr.shape[:2]
        s = self.cfg.min_side_px / max(1, min(h, w))
        if s > 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(text=[self.text], images=[img], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**inp).logits[0, -1].float()
        if self.cfg.mode == "color":
            ids = self.torch.tensor(self.color_ids, device=logits.device)
            probs = self.torch.softmax(logits[ids], dim=0)
            # merge colours that share a first token so P(target) is not double counted
            tgt = self.color_ids[self.color_names.index(self.target_color)]
            same = [k for k, i in enumerate(self.color_ids) if i == tgt]
            return float(probs[same[0]]) if len(same) == 1 else float(probs[same].sum() / len(same))
        two = self.torch.stack([logits[self.yes_id], logits[self.no_id]])
        return float(self.torch.softmax(two, dim=0)[0])


class VerifierWorker(threading.Thread):
    def __init__(self, cfg: VerifierCfg, verifier: VLMVerifier,
                 on_result: Callable[[VerifyResult], None], stop: threading.Event):
        super().__init__(name="tier2-vlm", daemon=True)
        self.cfg, self.v, self.on_result, self.stop = cfg, verifier, on_result, stop
        self.q: "queue.Queue[Candidate]" = queue.Queue(maxsize=cfg.queue_size)
        self.calls = 0
        self.busy = threading.Event()

    def submit(self, cand: Candidate) -> bool:
        """Non-blocking; returns False (candidate dropped) if the VLM is saturated."""
        try:
            self.q.put_nowait(cand)
            return True
        except queue.Full:
            log.debug("VLM queue full; dropping %s candidate from frame %d", cand.event, cand.frame_idx)
            return False

    def idle(self) -> bool:
        return self.q.empty() and not self.busy.is_set()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                cand = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.busy.set()
            try:
                t0 = time.perf_counter()
                p = self.v.p_yes(cand.context_bgr)
                lat = (time.perf_counter() - t0) * 1000.0
                self.calls += 1
                log.info("VLM[%s f%d] p_yes=%.2f (%.0f ms) tier1=%s", cand.event, cand.frame_idx, p, lat,
                         {k: round(x, 2) for k, x in cand.probs.items()})
                self.on_result(VerifyResult(cand, p, p >= self.cfg.p_yes_threshold, lat))
            except Exception:
                log.exception("VLM verification failed")
            finally:
                self.busy.clear()
