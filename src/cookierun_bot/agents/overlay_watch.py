from __future__ import annotations
import argparse
from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..config import load_config
from ..device import open_device
from ..gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE
from ..policies.rule_based import ActionDecision, StreamingRuleBasedAgent
from .action_watch import action_name


@dataclass(frozen=True)
class OverlayState:
    text: str
    bg: str
    fg: str
    visible: bool


SKILL_STATE = OverlayState("SKILL", "#7c3aed", "#ffffff", True)


def state_for_action(action: int, show_noop: bool = False) -> OverlayState:
    if action == ACTION_JUMP:
        return OverlayState("JUMP", "#ffb000", "#111111", True)
    if action == ACTION_SLIDE:
        return OverlayState("SLIDE", "#00c2ff", "#07131a", True)
    if show_noop:
        return OverlayState("READY", "#1f2933", "#e5e7eb", True)
    return OverlayState("", "#000000", "#ffffff", False)


def channeling_score(frame, cfg) -> float:
    zone = cfg.regions["play_area"].crop(frame)
    if zone.size == 0:
        return 0.0
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 70, 130]), np.array([45, 255, 255]))
    sparkle = cv2.inRange(hsv, np.array([0, 0, 225]), np.array([179, 80, 255]))
    return float(((yellow > 0) | (sparkle > 0)).mean())


def state_for_frame(frame, cfg, agent, show_noop: bool = False,
                    channeling_threshold: float = 0.18,
                    show_channeling: bool = True,
                    decision: ActionDecision | None = None) -> OverlayState:
    # ponytail: channeling is a visual suppression layer, not a game-specific state machine.
    if show_channeling and channeling_score(frame, cfg) >= channeling_threshold:
        return SKILL_STATE
    if decision is None:
        if hasattr(agent, "decide"):
            decision = agent.decide(frame)
        else:
            decision = ActionDecision(agent.act(frame), "legacy")
    return state_for_action(decision.action, show_noop)


class WarningLatch:
    def __init__(self, hold_ms: int = 220):
        self._hold_s = max(0, hold_ms) / 1000.0
        self._state = state_for_action(ACTION_NOOP)
        self._until = 0.0

    def update(self, candidate: OverlayState, now: float | None = None) -> OverlayState:
        now = time.monotonic() if now is None else now
        if candidate == SKILL_STATE:
            self._state = candidate
            self._until = now + self._hold_s
            return candidate
        if candidate.visible:
            self._state = candidate
            self._until = now + self._hold_s
            return candidate
        if self._state.visible and now < self._until:
            return self._state
        self._state = candidate
        return candidate


def make_clickthrough(root) -> None:
    """Best-effort Windows click-through overlay. No-op outside Windows/Tk support."""
    try:
        import ctypes
        hwnd = root.winfo_id()
        user32 = ctypes.windll.user32
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        gwl_exstyle = -20
        ws_ex_layered = 0x00080000
        ws_ex_transparent = 0x00000020
        ws_ex_topmost = 0x00000008
        style = get_style(hwnd, gwl_exstyle)
        set_style(hwnd, gwl_exstyle, style | ws_ex_layered | ws_ex_transparent | ws_ex_topmost)
    except Exception:
        pass


class OverlayApp:
    def __init__(self, cfg_path: str, interval_ms: int = 100,
                 show_noop: bool = False, geometry: str = "360x120+80+80",
                 hold_ms: int = 220, channeling_threshold: float = 0.18,
                 show_channeling: bool = True):
        import tkinter as tk

        self.cfg = load_config(cfg_path)
        self.device = open_device(self.cfg)
        self.agent = StreamingRuleBasedAgent(self.cfg)
        self.interval_ms = interval_ms
        self.show_noop = show_noop
        self.channeling_threshold = channeling_threshold
        self.show_channeling = show_channeling
        self.latch = WarningLatch(hold_ms)
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.82)
        self.root.geometry(geometry)
        self.root.configure(bg="#000000")
        self.label = tk.Label(
            self.root,
            text="STARTING",
            font=("Segoe UI", 42, "bold"),
            bd=0,
            padx=24,
            pady=16,
        )
        self.label.pack(expand=True, fill="both")
        make_clickthrough(self.root)

    def set_state(self, state: OverlayState) -> None:
        if not state.visible:
            self.root.withdraw()
            return
        self.root.deiconify()
        self.label.configure(text=state.text, bg=state.bg, fg=state.fg)
        self.root.configure(bg=state.bg)

    def tick(self) -> None:
        frame = self.device.last_frame()
        if frame is None:
            self.set_state(OverlayState("NO FRAME", "#7f1d1d", "#ffffff", True))
        else:
            decision = self.agent.decide(frame)
            candidate = state_for_frame(
                frame,
                self.cfg,
                self.agent,
                show_noop=self.show_noop,
                channeling_threshold=self.channeling_threshold,
                show_channeling=self.show_channeling,
                decision=decision,
            )
            self.set_state(self.latch.update(candidate))
            self.root.title(f"CookieRun Advisor: {action_name(decision.action)} {decision.reason}")
        self.root.after(self.interval_ms, self.tick)

    def run(self) -> None:
        self.device.start()
        self.agent.reset()
        try:
            self.tick()
            self.root.mainloop()
        finally:
            self.device.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Click-through read-only jump/slide overlay. Sends no input."
    )
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--hold-ms", type=int, default=220)
    parser.add_argument("--channeling-threshold", type=float, default=0.18)
    parser.add_argument("--no-channeling", action="store_true")
    parser.add_argument("--show-noop", action="store_true")
    parser.add_argument("--geometry", default="360x120+80+80")
    args = parser.parse_args(argv)
    OverlayApp(
        args.config,
        interval_ms=args.interval_ms,
        show_noop=args.show_noop,
        geometry=args.geometry,
        hold_ms=args.hold_ms,
        channeling_threshold=args.channeling_threshold,
        show_channeling=not args.no_channeling,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
