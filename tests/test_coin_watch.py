import numpy as np

from cookierun_bot.agents import coin_watch
from cookierun_bot.config import Config, Gestures, Region, RewardWeights


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "adb", 15, 8, "Episode 1",
                  {k: r for k in ["play_area", "coin_counter", "mystery_box_counter",
                                  "results_coins", "results_ingredients"]},
                  Gestures((1, 2), (3, 4), 300), RewardWeights(1, 50, 0.01, 10),
                  ["start"], ["buy"], "templates")


def test_coins_per_hour_uses_positive_delta_only():
    assert coin_watch.coins_per_hour(100, 125, 60.0) == 1500.0
    assert coin_watch.coins_per_hour(100, 90, 60.0) == 0.0
    assert coin_watch.coins_per_hour(None, 125, 60.0) == 0.0


def test_watch_device_never_sends_input(monkeypatch, fake_device):
    fake_device.set_frame(np.zeros((20, 20, 3), np.uint8))
    coin_values = iter([100, 100, 125])
    times = iter([0.0, 0.0, 60.0])
    lines = []

    monkeypatch.setattr(coin_watch, "read_coins", lambda frame, cfg: next(coin_values))
    monkeypatch.setattr(
        coin_watch,
        "read_results",
        lambda frame, cfg: {"coins": 0, "ingredients": 0},
    )

    coin_watch.watch_device(
        fake_device,
        _cfg(),
        frames=2,
        interval_s=0,
        stable_reads=1,
        now=lambda: next(times),
        sleep=lambda _: None,
        out=lines.append,
    )

    assert fake_device.taps == []
    assert fake_device.holds == []
    assert "coins/hr=1500" in lines[-1]


def test_format_sample_marks_unknown_coin_count():
    sample = coin_watch.CoinWatchSample(1, 0.0, None, "unknown", 0, 0.0, 12, 1)
    assert "coins=?" in coin_watch.format_sample(sample)


def test_read_sample_uses_results_when_live_counter_unreadable(monkeypatch, blank_frame):
    monkeypatch.setattr(coin_watch, "read_coins", lambda frame, cfg: None)
    monkeypatch.setattr(
        coin_watch,
        "read_results",
        lambda frame, cfg: {"coins": 20461, "ingredients": 0},
    )

    sample = coin_watch.read_sample(blank_frame, _cfg(), 1, None, 0.0)

    assert sample.coins == 20461
    assert sample.source == "results"
