from __future__ import annotations
from collections import deque
import cv2
import numpy as np


def preprocess(frame, play_area, size=(84, 84)) -> np.ndarray:
    crop = play_area.crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.uint8)


class FrameStack:
    def __init__(self, k: int = 4):
        self.k = k
        self._frames: deque = deque(maxlen=k)

    def reset(self, frame) -> np.ndarray:
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return np.stack(self._frames, axis=0)

    def push(self, frame) -> np.ndarray:
        self._frames.append(frame)
        return np.stack(self._frames, axis=0)
