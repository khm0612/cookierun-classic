"""M4.2 — wrap any policy with a learned pit-fall trigger.

M1.1 forensics proved the base policy is BLIND to the pits that kill it (41/46 no-jump falls
have jump-conf ~0). M4.1 proved a small detector recovers 75% of held-out pits from the same
pixels. This wrapper runs that detector (scripts/train_hazard.py's `hazard.pt`) every frame and,
when P(pit) crosses a threshold OUTSIDE bonustime, FORCES a jump — because the base contributes
~0 confidence, a soft gate-nudge can't lift it over the line; the trigger has to fire on its own.

Fully decoupled: it keeps its OWN K-ring of preprocessed frames (never touches the inner
agent's buffers) and only overrides the action, so with no AIFARM_HAZARD env set nothing here
runs and the deployed behaviour is byte-identical. Bonustime is detected via the inner hybrid's
reason prefix ("bonus/...") so the clean film dodger keeps the BONUSTIME gauntlets to itself.
"""
from __future__ import annotations
import dataclasses
import json
import os
import time
from collections import deque

import numpy as np
import cv2

from ..gestures import ACTION_JUMP
from .learned import build_convs


class HazardTrigger:
    def __init__(self, inner, hazard_path: str, meta_path: str, thr: float = 0.7,
                 cooldown_s: float = 0.25, max_per_episode: int = 2, check_every: int = 3):
        import torch
        self._torch = torch
        self.inner = inner
        meta = json.load(open(meta_path))
        self.K, self.H, self.W = int(meta["K"]), int(meta["H"]), int(meta["W"])
        self._crop = meta.get("crop", [0.1, 0.2, 1.0, 0.9])
        self._thr = float(thr)
        self._cd_s = float(cooldown_s)
        self._max_ep = int(max_per_episode)
        # LIVE FINDING (2026-07-14): running the head EVERY frame dropped fps 50->37, and low
        # fps is itself the dominant fall driver — a net loss. Throttle to every Nth frame; a
        # ~1.5s pit-approach window is ~50 frames, so every 3rd still gives ~17 looks.
        self._check_every = max(1, int(check_every))
        self._fcount = 0
        self._device = getattr(inner, "_device", torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"))
        self._net = _HazardNet(torch, meta).to(self._device).eval()
        self._net.load_state_dict(torch.load(hazard_path, map_location="cpu"))
        self._buf: deque = deque(maxlen=self.K)
        self._cd_until = 0.0
        self._ep_hits = 0          # jumps fired in the current sustained-P episode
        self._below = 0            # consecutive sub-thr frames (resets the episode)
        self.fires = 0             # total forced jumps this run (for the log)
        # ASYNC (2026-07-14 fix): the sync per-frame forward dropped fps 50->37 and low fps
        # is itself the dominant fall driver, so the detector was net-harmful. Run inference on
        # a BACKGROUND thread: the decision loop only stores the latest frame (a ref) and reads
        # a cached probability (a float) — both near-zero cost, so fps is unaffected. The worker
        # preprocesses + infers as fast as the GPU allows and publishes self._p_pit.
        import threading
        self._async = os.environ.get("AIFARM_HAZARD_ASYNC", "1") == "1"
        self._latest = None        # newest raw frame (written by decide, read by worker)
        self._p_pit = 0.0          # newest P(pit) (written by worker, read by decide)
        self._run_worker = self._async
        if self._async:
            self._wthread = threading.Thread(target=self._worker, daemon=True)
            self._wthread.start()
        print(f"[hazard] trigger armed: thr={self._thr} cd={self._cd_s}s "
              f"max/episode={self._max_ep} async={self._async}", flush=True)

    def _worker(self):
        """Background inference: preprocess the latest frame into the ring, run the head, publish
        P(pit). All hazard cost lives here, OFF the decision loop, so fps is untouched."""
        torch = self._torch
        last_proc = None
        while self._run_worker:
            fr = self._latest
            if fr is None or fr is last_proc:
                time.sleep(0.008)            # no new frame yet — don't stack duplicates
                continue
            last_proc = fr
            try:
                self._buf.append(self._preprocess(fr))
                buf = list(self._buf)
                if not buf:
                    continue
                while len(buf) < self.K:
                    buf.insert(0, buf[0])
                x = torch.from_numpy(np.stack(buf, 0)[None]).to(self._device).float().div_(255.0)
                with torch.no_grad():
                    self._p_pit = float(torch.sigmoid(self._net(x)).item())
            except Exception:
                time.sleep(0.01)

    # ---- interface parity with LearnedAgent / HybridPhaseAgent ----
    @property
    def explore(self):
        return self.inner.explore

    @explore.setter
    def explore(self, v):
        self.inner.explore = v

    def reset(self) -> None:
        self.inner.reset()
        self._buf.clear()
        self._latest = None        # don't let the worker infer on a stale prev-run frame
        self._p_pit = 0.0
        self._cd_until = 0.0
        self._ep_hits = 0
        self._below = 0
        self._fcount = 0
        self.fires = 0

    def observe(self, frame) -> None:
        self.inner.observe(frame)
        if self._async:
            self._latest = frame
        else:
            self._push(frame)

    def act(self, frame) -> int:
        return self.decide(frame).action

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._crop
        band = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def _push(self, frame):
        self._buf.append(self._preprocess(frame))

    def _infer_p_pit(self):
        buf = list(self._buf)
        while len(buf) < self.K:                 # oldest-frame pad on a cold ring
            buf.insert(0, buf[0])
        torch = self._torch
        x = torch.from_numpy(np.stack(buf, 0)[None]).to(self._device).float().div_(255.0)
        with torch.no_grad():
            return float(torch.sigmoid(self._net(x)).item())

    def decide(self, frame):
        d = self.inner.decide(frame)             # keeps the inner policy + its buffers warm
        if self._async:
            self._latest = frame                 # publish to the worker (cheap ref store)
        else:
            self._push(frame)
        # bonustime belongs to the film dodger — the hybrid tags those frames "bonus/..."
        if str(getattr(d, "reason", "")).startswith("bonus"):
            self._below += 1
            if self._below >= 3:
                self._ep_hits = 0
            return d
        if self._async:
            p = self._p_pit                      # read the worker's cached probability (a float)
        else:
            # SYNC fallback: throttle the head's forward pass every Nth frame (fps protection)
            self._fcount += 1
            if self._fcount % self._check_every != 0:
                return d
            p = self._infer_p_pit()
        if p < self._thr:
            self._below += 1
            if self._below >= 3:                 # a lull ends the episode -> re-arm the budget
                self._ep_hits = 0
            return d
        self._below = 0
        now = time.monotonic()
        if now < self._cd_until or self._ep_hits >= self._max_ep or d.action == ACTION_JUMP:
            return d
        self._cd_until = now + self._cd_s
        self._ep_hits += 1
        self.fires += 1
        return dataclasses.replace(d, action=ACTION_JUMP, reason=f"hazard:jump:{p:.2f}")


class _HazardNet:
    """Rebuilt to match scripts/train_hazard.py (conv trunk + Flatten->128->1)."""
    def __new__(cls, torch, meta):
        import torch.nn as nn
        convs, c, h, w = build_convs(nn, meta)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = convs
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * h * w, 128),
                                          nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 1))

            def forward(self, x):
                return self.head(self.convs(x)).squeeze(1)

        return Net()
