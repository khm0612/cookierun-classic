"""Imitation-learned dodge policy: a CNN (trained by behavioral cloning on the user's own
play, scripts/train2.py) predicts jump / slide / none from a stack of recent frames.
Drop-in for the rule-based agent — exposes .decide(frame) -> ActionDecision and .act(frame).

Everything that must match training EXACTLY (architecture, crop, input size, frame-stack
temporal spacing) is driven by model_meta.json, so training and inference cannot drift.
Loaded lazily so importing this module never requires torch unless a LearnedAgent is built.
"""
from __future__ import annotations
from collections import deque
import json
import time
import cv2
import numpy as np

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from .rule_based import ActionDecision

_ACTION = {"none": ACTION_NOOP, "jump": ACTION_JUMP, "slide": ACTION_SLIDE}


def build_net_from_meta(torch, meta):
    """Conv stack from meta['conv'] = [(out_ch, kernel, stride), ...] on K stacked grayscale
    frames, flattened WITH spatial layout preserved (no global pooling — obstacle POSITION
    is the signal), then fc -> 3 classes. Shared by train2.py and LearnedAgent."""
    import torch.nn as nn
    layers, in_ch = [], meta["K"]
    h, w = meta["H"], meta["W"]
    for out_ch, k, s in meta["conv"]:
        layers += [nn.Conv2d(in_ch, out_ch, k, s, k // 2), nn.ReLU()]
        in_ch = out_ch
        h, w = (h + s - 1) // s, (w + s - 1) // s
    layers += [nn.Flatten(), nn.Linear(in_ch * h * w, meta["fc"]), nn.ReLU(),
               nn.Dropout(0.3), nn.Linear(meta["fc"], len(meta["classes"]))]
    return nn.Sequential(*layers)


class LearnedAgent:
    """CNN behavioral-cloning policy. `conf` = minimum softmax probability to act (below it
    -> NOOP): trades a few missed dodges for far fewer spurious ones.

    `conf_slide` is much stricter: the two mistakes are NOT symmetric. A wrong jump just
    lands back (worst case an HP hit that potions heal); a wrong slide keeps the cookie LOW
    through a platform gap = pit death (observed live: the model mistook a jump-onto ledge
    for a slide-under obstacle and slid into the pit). With only 29 slide examples in the
    demo the slide head is the least-trusted output, so it must be near-certain to act."""

    def __init__(self, cfg, model_path: str, meta_path: str, conf: float = 0.6,
                 conf_slide: float = 0.90):
        import torch
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.load(open(meta_path))
        self.meta = meta
        self.K, self.H, self.W = meta["K"], meta["H"], meta["W"]
        self.classes = meta["classes"]
        self._crop = meta.get("crop", [0.0, 0.0, 1.0, 1.0])
        # stack frames at the TRAINING fps spacing: live capture may run far faster (dxcam
        # ~270fps) and a 15ms-span stack would be out-of-distribution. meta['fps'] is now
        # MEASURED from the demos' real frame cadence at train time (scripts/train2.py), so
        # it tracks the recorder instead of a stale hardcoded assumption.
        self._frame_gap = 1.0 / meta.get("fps", 35.0)
        self._last_stacked = 0.0
        self._conf = conf
        self._conf_slide = conf_slide
        self._buf: deque = deque(maxlen=self.K)
        self._net = build_net_from_meta(torch, meta)
        self._net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._net.to(self._device).eval()
        # TIME-based jump cooldown (tick-based breaks at 100+fps live loops). Small on
        # purpose: the human demo double-jumps with gaps down to ~0.12s (p5 0.20s), so a
        # long cooldown blocks half their real dodge pattern; the device's one-finger
        # throttle already absorbs per-frame refires of the same decision.
        self._jump_cd_s = 0.25
        self._cd_until = 0.0
        # Slide gets its OWN cooldown ~= its hold duration. play_until_death re-decides per
        # frame and each SLIDE re-issues device.hold(...slide_hold_ms) fire-and-forget, so an
        # ungated high-conf slide at 60-270fps queues many overlapping holds = seconds of
        # continuous LOW posture (over-slides through a platform gap = pit death) + an adb
        # backlog. Don't re-issue a slide while the previous hold is still in flight.
        self._slide_cd_s = getattr(getattr(cfg, "gestures", None), "slide_hold_ms", 500) / 1000.0
        self._slide_cd_until = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._cd_until = 0.0
        self._slide_cd_until = 0.0
        self._last_stacked = 0.0

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._crop
        band = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def _stack(self, frame):
        now = time.monotonic()
        if not self._buf or now - self._last_stacked >= self._frame_gap:
            self._buf.append(self._preprocess(frame).astype(np.float32) / 255.0)
            self._last_stacked = now
        else:                                   # too soon: refresh only the newest slot
            self._buf[-1] = self._preprocess(frame).astype(np.float32) / 255.0
        while len(self._buf) < self.K:
            self._buf.appendleft(self._buf[0])
        return np.stack(self._buf, 0)[None]     # (1,K,H,W)

    def decide(self, frame) -> ActionDecision:
        x = self._torch.from_numpy(self._stack(frame)).to(self._device)
        with self._torch.no_grad():
            p = self._torch.softmax(self._net(x)[0], 0).cpu().numpy()
        i = int(p.argmax())
        cls = self.classes[i]
        action = _ACTION[cls]
        gate = self._conf_slide if action == ACTION_SLIDE else self._conf
        if action == ACTION_NOOP or p[i] < gate:
            return ActionDecision(ACTION_NOOP, f"model:{cls}:{p[i]:.2f}")
        now = time.monotonic()
        if action == ACTION_JUMP:
            if now < self._cd_until:
                return ActionDecision(ACTION_NOOP, "model:jump-cooldown")
            self._cd_until = now + self._jump_cd_s
        elif action == ACTION_SLIDE:
            if now < self._slide_cd_until:
                return ActionDecision(ACTION_NOOP, "model:slide-cooldown")
            self._slide_cd_until = now + self._slide_cd_s
        return ActionDecision(action, f"model:{cls}:{p[i]:.2f}")

    def act(self, frame) -> int:
        return self.decide(frame).action
