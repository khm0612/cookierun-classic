"""Phase-aware two-model policy: an AGGRESSIVE earner in normal stages, the CLEAN film
dodger during BONUSTIME.

WHY: the 2026-07-12 A/Bs proved a stable trade — plain_hf4 (champion arch + deep demo)
earns the most coins via jump-spam, while sslfilm_hf4 plays ~35% cleaner and survives the
pit-heavy BONUSTIME platform gauntlets (where every model death occurs) once its jump gate
is opened. This wrapper routes each phase to the model that wins it, switching on the same
BONUSTIME banner detector + latch the diagnostics use (condition.py). The latch's 3s decay
is the switch hysteresis — no thrashing on the banner's pulse.

Both agents' frame stacks are kept warm every frame (observe()) so a switch never hands
control to a model with a stale/degenerate K-stack; only the ACTIVE model runs inference.
Interface-compatible with LearnedAgent (decide/act/reset/explore) so farm.play_until_death
and ai_farm need no changes beyond construction.
"""
from __future__ import annotations
import dataclasses
import time

from .condition import bonustime_bgr, load_bonus_template, BONUS_LATCH_S


class HybridPhaseAgent:
    def __init__(self, base_agent, bonus_agent, templates_dir="templates",
                 latch_s: float = BONUS_LATCH_S, check_s: float = 0.25):
        self.base = base_agent
        self.bonus = bonus_agent
        self._tpl = load_bonus_template(templates_dir)
        if self._tpl is None:
            print("[hybrid] WARNING: bonustime_norm.png missing — bonus model will never "
                  "activate (base model runs everything)")
        self._latch_s = float(latch_s)
        self._check_s = float(check_s)
        self._seen = -1e9
        self._next_check = 0.0
        self._active_name = "base"

    @property
    def _device(self):
        """ai_farm's startup banner prints agent._device — delegate to the base model."""
        return getattr(self.base, "_device", "?")

    # farm code sets .explore on the agent; fan it out to both members
    @property
    def explore(self):
        return self.base.explore

    @explore.setter
    def explore(self, v):
        self.base.explore = v
        self.bonus.explore = v

    def reset(self) -> None:
        self.base.reset()
        self.bonus.reset()
        self._seen = -1e9
        self._next_check = 0.0
        self._active_name = "base"

    def _bonus_active(self, frame, now: float) -> bool:
        if now >= self._next_check:
            self._next_check = now + self._check_s
            if bonustime_bgr(frame, self._tpl):
                self._seen = now
        return now - self._seen < self._latch_s

    def decide(self, frame):
        now = time.monotonic()
        in_bonus = self._bonus_active(frame, now)
        active, passive = (self.bonus, self.base) if in_bonus else (self.base, self.bonus)
        name = "bonus" if in_bonus else "base"
        if name != self._active_name:
            self._active_name = name
        passive.observe(frame)              # keep the idle model's K-stack warm
        d = active.decide(frame)
        # tag the source so live logs show which model fired. Prefix must not contain ':'
        # — ai_farm parses reason.split(":")[1] as the class name.
        return dataclasses.replace(d, reason=f"{name}/{d.reason}")

    def act(self, frame) -> int:
        return self.decide(frame).action
