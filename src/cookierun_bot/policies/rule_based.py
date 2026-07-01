from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE

_DARK_FRACTION = 0.05       # >5% dark pixels in a band = obstacle. Tune on real footage.


@dataclass
class Features:
    low_obstacle: bool       # obstacle in the lower (ground) band -> jump
    overhead_obstacle: bool  # obstacle only in the upper band -> slide


def _dark_fraction(band) -> float:
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    return float((gray < 60).mean())


def extract_features(frame, cfg) -> Features:
    zone = cfg.regions["play_area"].crop(frame)
    h = zone.shape[0]
    upper = zone[0:h // 2]
    lower = zone[h // 2:]
    low = _dark_fraction(lower) > _DARK_FRACTION
    up = _dark_fraction(upper) > _DARK_FRACTION
    return Features(low_obstacle=low, overhead_obstacle=(up and not low))


class RuleBasedAgent:
    def __init__(self, cfg):
        self._cfg = cfg

    def reset(self) -> None:
        pass

    def act(self, frame) -> int:
        f = extract_features(frame, self._cfg)
        if f.low_obstacle:
            return ACTION_JUMP           # survive first: clear ground obstacles / gaps
        if f.overhead_obstacle:
            return ACTION_SLIDE
        return ACTION_NOOP               # run on; coins/jellies in the lane are auto-collected
