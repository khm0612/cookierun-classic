import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.menu import MenuNavigator


class StubMatcher:
    def __init__(self, present_names, points=None):
        self._present = set(present_names)
        self._points = points or {}
    def present(self, frame, name, threshold=0.8):
        return name in self._present
    def find(self, frame, name, threshold=0.8):
        return self._points.get(name)


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {k: r for k in ["play_area", "coin_counter", "mystery_box_counter",
                                  "results_coins", "results_ingredients"]},
                  Gestures((0, 0), (0, 0), 300), RewardWeights(1, 50, 0.01, 10),
                  ["restart", "ok"], ["buy", "revive_crystals"], "templates")


def test_denylist_blocks_tapping(fake_device):
    m = StubMatcher(present_names=["ok", "revive_crystals"], points={"ok": (5, 5)})
    nav = MenuNavigator(fake_device, m, _cfg())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert nav.is_spend_dialog(frame) is True
    assert nav.advance(frame) == "spend_blocked"
    assert fake_device.taps == []          # never tapped a spend dialog


def test_taps_first_allowlist_button(fake_device):
    m = StubMatcher(present_names=["restart"], points={"restart": (30, 40)})
    nav = MenuNavigator(fake_device, m, _cfg())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert nav.advance(frame) == "tapped"
    assert fake_device.taps == [(30, 40)]


def test_idle_when_nothing_present(fake_device):
    m = StubMatcher(present_names=[])
    nav = MenuNavigator(fake_device, m, _cfg())
    assert nav.advance(np.zeros((100, 100, 3), np.uint8)) == "idle"
