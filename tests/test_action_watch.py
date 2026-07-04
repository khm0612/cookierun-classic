import numpy as np

from cookierun_bot.agents import action_watch
from cookierun_bot.agents.coin_watch import CoinWatchSample
from cookierun_bot.config import Config, Gestures, Region, RewardWeights


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "adb", 15, 8, "Episode 1",
                  {"play_area": Region(0, 0, 100, 100), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((1, 2), (3, 4), 300), RewardWeights(1, 50, 0.01, 10),
                  ["start"], ["buy"], "templates")


def test_action_name_maps_known_actions():
    assert action_watch.action_name(0) == "noop"
    assert action_watch.action_name(1) == "jump"
    assert action_watch.action_name(2) == "slide"


def test_read_action_sample_reports_jump_without_input(monkeypatch, fake_device):
    frame = np.full((720, 1280, 3), 40, np.uint8)   # dark zone
    frame[470:580, 500:640] = (20, 120, 255)        # orange ground pumpkin...
    frame[510:530, 545:595] = (10, 10, 10)          # ...with a black face -> jump
    fake_device.set_frame(frame)

    monkeypatch.setattr(
        action_watch,
        "read_sample",
        lambda frame, cfg, frame_no, start_coins, elapsed_s: CoinWatchSample(
            frame_no, elapsed_s, 100, "live", 0, 0.0, 0, 0
        ),
    )

    lines = []
    action_watch.watch_device(
        fake_device,
        _cfg(),
        frames=1,
        interval_s=0,
        now=lambda: 0.0,
        sleep=lambda _: None,
        out=lines.append,
    )

    assert fake_device.taps == []
    assert fake_device.holds == []
    assert "action=noop" in lines[0]
    assert "reason=confirming:hazard:jump" in lines[0]


def test_read_action_sample_reports_jump_after_confirmation(monkeypatch):
    frame1 = np.full((720, 1280, 3), 40, np.uint8)
    frame1[470:580, 500:640] = (20, 120, 255)
    frame1[510:530, 545:595] = (10, 10, 10)
    frame2 = frame1.copy()

    class Stream:
        taps = []
        holds = []

        def __init__(self):
            self.frames = [frame1, frame2]

        def last_frame(self):
            return self.frames.pop(0)

    monkeypatch.setattr(
        action_watch,
        "read_sample",
        lambda frame, cfg, frame_no, start_coins, elapsed_s: CoinWatchSample(
            frame_no, elapsed_s, 100, "live", 0, 0.0, 0, 0
        ),
    )
    lines = []

    action_watch.watch_device(
        Stream(),
        _cfg(),
        frames=2,
        interval_s=0,
        now=lambda: 0.0,
        sleep=lambda _: None,
        out=lines.append,
    )

    assert "reason=confirming:hazard:jump" in lines[0]
    assert "action=jump" in lines[1]
    assert "reason=hazard:jump" in lines[1]


def test_format_action_sample_includes_features():
    sample = action_watch.ActionWatchSample(
        1, 0.0, 2, "slide", "hazard:slide", 2, False, "slide",
        False, True, False, "[frame 1]",
    )
    out = action_watch.format_action_sample(sample)
    assert "action=slide" in out
    assert "reason=hazard:slide" in out
    assert "overhead:True" in out


def test_read_action_sample_suppresses_actions_when_not_in_run(monkeypatch):
    class Matcher:
        def present(self, frame, name, threshold=0.8):
            return False

    frame = np.full((720, 1280, 3), 40, np.uint8)

    sample = action_watch.read_action_sample(
        frame,
        _cfg(),
        agent=object(),
        frame_no=1,
        start_coins=None,
        elapsed_s=0.0,
        matcher=Matcher(),
    )

    assert sample.action_name == "noop"
    assert sample.reason == "not-in-run"
