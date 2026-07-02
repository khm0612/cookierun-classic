"""Unattended farm loop for CookieRun Classic on LDPlayer (guest coords 2560x1440).

Flow (Episode 1): menu Play -> pre-run boost screen Play -> run -> rule-based agent
plays until death (screen stops scrolling) -> Result screen -> OK -> back to menu -> repeat.

Currency guardrail: after death we WAIT out any 'revive with crystals' prompt (it has a
countdown and auto-dismisses) before tapping OK, and we never tap buy/revive spots.
"""
from __future__ import annotations
import sys
import time

import numpy as np

from .config import load_config
from .device import open_device
from .gestures import apply_action
from .metrics import Metrics, RunResult
from .policies.rule_based import RuleBasedAgent

# On-screen coords in guest space (2560x1440), calibrated for this LDPlayer + game.
MENU_PLAY = (1620, 1290)     # green Play! on the main menu
BOOST_PLAY = (1620, 1240)    # Play! on the pre-run boost screen (does NOT buy boosts)
JUMP = (360, 1220)           # jump zone (also dismisses the fast-start prompt)
RESULT_OK = (780, 1240)      # OK on the Result screen


def _diff(a, b) -> float:
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def _scrolling(dev, dt: float = 0.25, thresh: float = 8.0) -> bool:
    """A run is in progress iff the background is scrolling (big frame-to-frame diff)."""
    a = dev.last_frame()
    time.sleep(dt)
    b = dev.last_frame()
    return a is not None and b is not None and _diff(a, b) > thresh


def ensure_running(dev, tries: int = 5) -> bool:
    """Get from the menu (or a settled screen) into an actual run."""
    for _ in range(tries):
        if _scrolling(dev):
            return True
        dev.tap(*MENU_PLAY)
        time.sleep(2.8)
        dev.tap(*BOOST_PLAY)
        time.sleep(2.2)
        dev.tap(*JUMP)          # dismiss the "Tap to activate Fast Start Boost!" prompt
        time.sleep(1.5)
    return _scrolling(dev)


def play_until_death(dev, cfg, agent, max_s: float = 300.0) -> float:
    """Run the rule-based agent until the screen stops scrolling (death/results)."""
    agent.reset()
    prev = None
    still = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_s:
        f = dev.last_frame()
        apply_action(dev, agent.act(f), cfg.gestures)
        if prev is not None and _diff(f, prev) < 2.5:
            still += 1
            if still >= 8:       # ~8 near-identical frames => not running anymore
                break
        else:
            still = 0
        prev = f
        time.sleep(1.0 / cfg.decision_hz)
    return time.monotonic() - t0


def clear_result(dev) -> None:
    """Dismiss the Result screen back to the menu. Waits out any crystal-revive prompt
    first (guardrail: never tap revive/buy)."""
    time.sleep(8.0)              # let a 'revive?' countdown time out if one appeared
    for _ in range(4):
        if _scrolling(dev):
            return
        dev.tap(*RESULT_OK)
        time.sleep(2.0)


def farm(cfg_path: str = "config.yaml", max_runs: int | None = None) -> None:
    cfg = load_config(cfg_path)
    dev = open_device(cfg)
    dev.start()
    print("device ready; game-area", getattr(dev, "_ga", "?"), "guest", dev.resolution)
    agent = RuleBasedAgent(cfg)
    metrics = Metrics()
    run = 0
    try:
        while max_runs is None or run < max_runs:
            if not ensure_running(dev):
                print("!! could not start a run (unexpected screen) — stopping")
                break
            dur = play_until_death(dev, cfg, agent)
            time.sleep(1.5)
            clear_result(dev)
            run += 1
            metrics.add(RunResult(0, 0, dur))
            print(f"[run {run}] survived {dur:.0f}s | {metrics.summary()}")
    finally:
        dev.stop()
        print("FINAL:", metrics.summary())


if __name__ == "__main__":
    args = sys.argv[1:]
    farm(args[0] if args else "config.yaml",
         int(args[1]) if len(args) > 1 else None)
