import numpy as np
import pytest


class FakeDevice:
    """In-memory Device stand-in for tests."""
    def __init__(self, frame=None, resolution=(1080, 1920)):
        self._frame = frame
        self._resolution = resolution
        self.taps = []          # list of (x, y)
        self.holds = []         # list of (x, y, duration_ms)
        self.started = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def last_frame(self): return self._frame
    def set_frame(self, frame): self._frame = frame

    @property
    def resolution(self): return self._resolution

    def tap(self, x, y): self.taps.append((x, y))
    def hold(self, x, y, duration_ms): self.holds.append((x, y, duration_ms))


@pytest.fixture
def fake_device():
    return FakeDevice(frame=np.zeros((1920, 1080, 3), dtype=np.uint8))


@pytest.fixture
def blank_frame():
    return np.zeros((1920, 1080, 3), dtype=np.uint8)
