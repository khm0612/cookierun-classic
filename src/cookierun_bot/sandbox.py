from __future__ import annotations
import argparse
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .config import Region
from .gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE
from .policies.rule_based import RuleBasedAgent


DEFAULT_EVENTS = (
    "coin", "low", "coin", "overhead", "coin",
    "low", "coin", "overhead", "coin", "coin",
)


@dataclass(frozen=True)
class SandboxStats:
    ticks: int
    coins: int
    damage: int
    actions: tuple[int, ...]


def cfg():
    return SimpleNamespace(regions={"play_area": Region(0, 0, 100, 100)})


def frame_for(event: str) -> np.ndarray:
    # 720x1280 stand-in frame on a dark background; hazards are painted as the CLASSIFIED
    # kinds the agent acts on (see rule_based._orange_hazard): a face-gated orange ground
    # blob = pumpkin -> jump; a top-anchored orange column = trunk wall -> slide. "coin" =
    # clear track (the magnet collects; the disciplined agent does not jump for coins).
    frame = np.full((720, 1280, 3), 40, np.uint8)
    if event == "low":                        # ground pumpkin -> jump
        frame[470:580, 500:640] = (20, 120, 255)
        frame[510:530, 545:595] = (10, 10, 10)
    elif event == "overhead":                 # hanging trunk wall -> slide
        frame[0:400, 480:600] = (20, 120, 255)
        frame[150:260, 495:585] = (10, 10, 10)
    return frame


def damage_for(event: str, action: int) -> int:
    if event == "low" and action != ACTION_JUMP:
        return 1
    if event == "overhead" and action != ACTION_SLIDE:
        return 1
    return 0


def run(events=DEFAULT_EVENTS) -> SandboxStats:
    agent = RuleBasedAgent(cfg())
    agent.reset()
    coins = 0
    damage = 0
    actions = []
    for event in events:
        action = agent.act(frame_for(event))
        actions.append(action)
        damage += damage_for(event, action)
        if event == "coin":
            coins += 1
    return SandboxStats(len(tuple(events)), coins, damage, tuple(actions))


def action_name(action: int) -> str:
    return {
        ACTION_NOOP: "noop",
        ACTION_JUMP: "jump",
        ACTION_SLIDE: "slide",
    }.get(action, f"unknown:{action}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline runner sandbox.")
    parser.add_argument("events", nargs="*", default=list(DEFAULT_EVENTS))
    args = parser.parse_args(argv)
    stats = run(tuple(args.events))
    print(f"ticks={stats.ticks} coins={stats.coins} damage={stats.damage}")
    print("actions=" + ",".join(action_name(a) for a in stats.actions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
