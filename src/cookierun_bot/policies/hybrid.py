"""HybridAgent: the imitation-learned model (primary) + high-precision CV hazard overrides.

Live diagnosis of the 2-demo model (B_win25): it survives ~262s but ~89% of its hits were
logged `model:none` — it TANKS obstacles it does not recognise instead of dodging them.
This wraps the calibrated Episode-1 hazard detectors AROUND the model: when the model is
PASSIVE (NOOP), a high-precision detector can still fire the dodge —

  * `_pit_ahead` -> JUMP    : falling into a pit is INSTANT death (not a tankable HP nick),
                              so this is the single highest-value override.
  * `_hazard`=='slide' -> SLIDE : overhead trunk walls / scissors are NOT jumpable and the
                              model's slide head is its weakest output.
  * `_hazard`=='jump'  -> JUMP  : optional; OFF by default (the model already jumps plenty,
                              and the orange classifier can false-fire on coin/skill items).

When the model DOES act we trust it (it is the better general policy). The CV suite costs
~15ms, so it is THROTTLED to `cv_hz` to avoid dropping the model's own ~70fps frame rate,
and discrete JUMP overrides are cooldowned so a noisy detector cannot spam. The SLIDE
override is deliberately NOT cooldowned: it re-asserts every throttled tick while the hazard
persists so the SlideHold in play_until_death keeps the finger down through the whole
obstacle (the grace window bridges the throttle gaps).
"""
from __future__ import annotations
import time

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from .rule_based import ActionDecision, _pit_ahead, _hazard


class HybridAgent:
    def __init__(self, cfg, model_path: str | None = None, meta_path: str | None = None,
                 conf: float = 0.6, conf_slide: float = 0.90,
                 overrides=("pit", "slide"), cv_hz: float = 30.0,
                 override_cd_s: float = 0.35, learned=None):
        if learned is None:
            from .learned import LearnedAgent           # lazy: importing hybrid != needing torch
            learned = LearnedAgent(cfg, model_path, meta_path, conf=conf,
                                   conf_slide=conf_slide)
        self._learned = learned
        self._ov = set(overrides)
        self._cv_gap = 1.0 / cv_hz if cv_hz > 0 else 0.0
        self._cd_s = override_cd_s
        self._last_cv = 0.0
        self._cd_until = 0.0
        # passthrough so callers/logs that read agent._device (e.g. learned_check) still work
        self._device = getattr(learned, "_device", "cpu")

    def reset(self) -> None:
        self._learned.reset()
        self._last_cv = 0.0
        self._cd_until = 0.0

    def decide(self, frame, now: float | None = None) -> ActionDecision:
        d = self._learned.decide(frame)
        if d.action != ACTION_NOOP:
            return d                                   # the model acted -> trust it
        if now is None:
            now = time.monotonic()
        if now < self._cd_until or now - self._last_cv < self._cv_gap:
            return d                                   # cooldown / throttle: skip CV this tick
        self._last_cv = now
        if "pit" in self._ov and _pit_ahead(frame):
            self._cd_until = now + self._cd_s          # jump = discrete -> cooldown
            return ActionDecision(ACTION_JUMP, "cv:pit")
        if self._ov & {"jump", "slide"}:
            haz = _hazard(frame)
            if haz == "slide" and "slide" in self._ov:
                return ActionDecision(ACTION_SLIDE, "cv:slide")   # sustained -> no cooldown
            if haz == "jump" and "jump" in self._ov:
                self._cd_until = now + self._cd_s
                return ActionDecision(ACTION_JUMP, "cv:jump")
        return d

    def act(self, frame) -> int:
        return self.decide(frame).action
