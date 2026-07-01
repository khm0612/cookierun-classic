from __future__ import annotations
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .capture import preprocess, FrameStack
from .detect import detect_death, read_coins, read_mystery_boxes
from .gestures import apply_action, N_ACTIONS
from .reward import RewardTracker


class CookieRunEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, device, cfg, matcher, tick_sleep=None):
        super().__init__()
        self._device = device
        self._cfg = cfg
        self._matcher = matcher
        self._stack = FrameStack(k=4)
        self._reward = RewardTracker(cfg.reward)
        self._tick_sleep = tick_sleep or (lambda: time.sleep(1.0 / cfg.decision_hz))
        self.observation_space = spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._last_raw = None

    def _grab(self):
        frame = self._device.last_frame()
        if frame is None:
            pa = self._cfg.regions["play_area"]
            frame = np.zeros((pa.h + pa.y, pa.w + pa.x, 3), np.uint8)
        self._last_raw = frame
        return frame

    def last_raw_frame(self):
        return self._last_raw

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reward.reset()
        frame = self._grab()
        obs = self._stack.reset(preprocess(frame, self._cfg.regions["play_area"]))
        return obs, {"coins": 0, "boxes": 0, "dead": False}

    def step(self, action):
        apply_action(self._device, int(action), self._cfg.gestures)
        self._tick_sleep()
        frame = self._grab()
        dead = detect_death(frame, self._matcher)
        coins = read_coins(frame, self._cfg)
        boxes = read_mystery_boxes(frame, self._cfg)
        reward = self._reward.update(coins=coins, boxes=boxes, dead=dead)
        obs = self._stack.push(preprocess(frame, self._cfg.regions["play_area"]))
        info = {"coins": coins if coins is not None else 0, "boxes": boxes, "dead": dead}
        return obs, reward, dead, False, info

    def close(self):
        self._device.stop()
