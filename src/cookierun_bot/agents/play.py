from __future__ import annotations
import sys
import time

from ..config import load_config
from ..device import open_device
from ..detect import TemplateMatcher, read_results
from ..env import CookieRunEnv
from ..menu import MenuNavigator
from ..metrics import Metrics, RunResult
from ..policies.rule_based import RuleBasedAgent


def _drive_menu_until_running(nav, device, timeout=30.0):
    """Tap allowlist buttons until the run starts (or timeout). Never taps spend dialogs."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame = device.last_frame()
        if frame is None:
            time.sleep(0.2); continue
        state = nav.advance(frame)
        if state == "spend_blocked":
            time.sleep(0.5)          # wait for the spend dialog to be dismissed elsewhere
        elif state == "idle":
            return True              # nothing left to tap -> assume in-run
        time.sleep(0.4)
    return False


def play(cfg_path="config.yaml", max_runs=None) -> None:
    cfg = load_config(cfg_path)
    device = open_device(cfg)
    device.start()
    time.sleep(2.0)
    matcher = TemplateMatcher(cfg.templates_dir)
    env = CookieRunEnv(device, cfg, matcher)
    agent = RuleBasedAgent(cfg)
    nav = MenuNavigator(device, matcher, cfg)
    metrics = Metrics()

    run = 0
    try:
        while max_runs is None or run < max_runs:
            _drive_menu_until_running(nav, device)
            obs, _ = env.reset()
            agent.reset()
            t0 = time.monotonic()
            terminated = False
            while not terminated:
                frame = env.last_raw_frame()
                action = agent.act(frame) if frame is not None else 0
                obs, reward, terminated, truncated, info = env.step(action)
            duration = time.monotonic() - t0
            results = read_results(device.last_frame(), cfg)
            metrics.add(RunResult(results["coins"], results["ingredients"], duration))
            run += 1
            print(f"[run {run}] {results} dur={duration:.1f}s | {metrics.summary()}")
            # collect rewards + replay, guardrail-protected
            _drive_menu_until_running(nav, device)
    finally:
        env.close()
        print("FINAL:", metrics.summary())


if __name__ == "__main__":
    args = sys.argv[1:]
    play(args[0] if args else "config.yaml")
