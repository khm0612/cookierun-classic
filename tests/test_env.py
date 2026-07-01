import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.env import CookieRunEnv


class StubMatcher:
    def __init__(self, dead=False): self.dead = dead
    def present(self, frame, name, threshold=0.8):
        return self.dead and name in ("results", "gameover")
    def find(self, frame, name, threshold=0.8): return None


def _cfg():
    r = Region(0, 0, 100, 100)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {"play_area": Region(0, 0, 200, 200), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((10, 20), (30, 40), 300),
                  RewardWeights(1, 50, 0.01, 10), ["ok"], ["buy"], "templates")


def test_reset_returns_stacked_obs(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(), tick_sleep=lambda: None)
    obs, info = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8


def test_step_jump_taps_and_returns_five_tuple(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(), tick_sleep=lambda: None)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)   # jump
    assert fake_device.taps == [(10, 20)]
    assert obs.shape == (4, 84, 84)
    assert terminated is False
    assert set(["coins", "boxes", "dead"]).issubset(info)


def test_step_terminates_on_death(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(dead=True), tick_sleep=lambda: None)
    env.reset()
    _, reward, terminated, _, info = env.step(0)
    assert terminated is True and info["dead"] is True
    assert reward < 0            # death penalty dominates
