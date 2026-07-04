from __future__ import annotations
import argparse
import time
from dataclasses import dataclass

from ..config import load_config
from ..detect import TemplateMatcher
from ..device import open_device
from ..gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE
from ..policies.rule_based import ActionDecision, StreamingRuleBasedAgent, extract_features
from .coin_watch import format_sample, read_sample


_ACTION_NAMES = {
    ACTION_NOOP: "noop",
    ACTION_JUMP: "jump",
    ACTION_SLIDE: "slide",
}


@dataclass(frozen=True)
class ActionWatchSample:
    frame_no: int
    elapsed_s: float
    action: int
    action_name: str
    reason: str
    confirmed: int
    pit_ahead: bool
    hazard: str | None
    low_obstacle: bool
    overhead_obstacle: bool
    coins_ahead: bool
    coin_line: str


def action_name(action: int) -> str:
    return _ACTION_NAMES.get(action, f"unknown:{action}")


def read_action_sample(frame, cfg, agent, frame_no: int,
                       start_coins: int | None, elapsed_s: float,
                       include_coins: bool = False,
                       matcher: TemplateMatcher | None = None) -> ActionWatchSample:
    if matcher is not None and not matcher.present(frame, "slide", 0.72):
        decision = ActionDecision(ACTION_NOOP, "not-in-run", extract_features(frame, cfg))
    elif hasattr(agent, "decide"):
        decision = agent.decide(frame)
    else:
        decision = ActionDecision(agent.act(frame), "legacy", extract_features(frame, cfg))
    features = decision.features or extract_features(frame, cfg)
    coin_line = f"[frame {frame_no}] t={elapsed_s:.3f}s"
    if include_coins:
        coins = read_sample(frame, cfg, frame_no, start_coins, elapsed_s)
        coin_line = format_sample(coins)
    return ActionWatchSample(
        frame_no=frame_no,
        elapsed_s=elapsed_s,
        action=decision.action,
        action_name=action_name(decision.action),
        reason=decision.reason,
        confirmed=decision.confirmed,
        pit_ahead=features.pit_ahead,
        hazard=features.hazard,
        low_obstacle=features.low_obstacle,
        overhead_obstacle=features.overhead_obstacle,
        coins_ahead=features.coins_ahead,
        coin_line=coin_line,
    )


def format_action_sample(s: ActionWatchSample) -> str:
    # ponytail: advisor only; it deliberately reports the action without executing it.
    return (
        f"{s.coin_line} action={s.action_name} reason={s.reason} "
        f"confirmed={s.confirmed} "
        f"features={{pit:{s.pit_ahead}, hazard:{s.hazard}, low:{s.low_obstacle}, "
        f"overhead:{s.overhead_obstacle}, coins:{s.coins_ahead}}}"
    )


def watch_device(device, cfg, frames: int | None = None, interval_s: float = 1.0 / 60.0,
                 include_coins: bool = False, now=time.monotonic,
                 sleep=time.sleep, out=print, matcher: TemplateMatcher | None = None) -> None:
    agent = StreamingRuleBasedAgent(cfg)
    agent.reset()
    start_time = now()
    start_coins: int | None = None
    frame_no = 0

    while frames is None or frame_no < frames:
        frame_no += 1
        frame = device.last_frame()
        if frame is None:
            out(f"[frame {frame_no}] no frame")
        else:
            elapsed_s = max(0.0, now() - start_time)
            sample = read_action_sample(
                frame, cfg, agent, frame_no, start_coins, elapsed_s,
                include_coins=include_coins, matcher=matcher,
            )
            out(format_action_sample(sample))

        if frames is None or frame_no < frames:
            sleep(interval_s)


def watch(cfg_path: str = "config.yaml", frames: int | None = None,
          interval_s: float = 1.0 / 60.0, include_coins: bool = False) -> None:
    cfg = load_config(cfg_path)
    device = open_device(cfg)
    device.start()
    try:
        matcher = TemplateMatcher(cfg.templates_dir)
        watch_device(device, cfg, frames=frames, interval_s=interval_s,
                     include_coins=include_coins, matcher=matcher)
    finally:
        device.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only action advisor. Prints jump/slide/noop; sends no input."
    )
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--coins", action="store_true")
    args = parser.parse_args(argv)
    interval = args.interval if args.interval is not None else 1.0 / max(args.hz, 1.0)
    watch(args.config, frames=args.frames, interval_s=interval, include_coins=args.coins)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
