from dataclasses import dataclass

import numpy as np

from cookierun_bot.config import SpendingConfig
from cookierun_bot import farm
from cookierun_bot import farm_common
from cookierun_bot.gift_draw import GiftDrawResult


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class FakeDevice:
    def __init__(self, frames):
        self._frames = list(frames)
        self.calls = 0

    def last_frame(self):
        self.calls += 1
        if len(self._frames) > 1:
            return self._frames.pop(0)
        return self._frames[0] if self._frames else None


class AnimatedOkDevice:
    def __init__(self):
        self.taps = []
        self._running = False
        self._tick = 0

    def last_frame(self):
        if self._running:
            return np.full((4, 4, 3), 7, np.uint8)
        self._tick += 1
        frame = np.full((4, 4, 3), self._tick * 8, np.uint8)
        frame[0, 0, 0] = 255
        return frame

    def tap(self, x, y):
        self.taps.append((x, y))
        self._running = True


class PopupDevice:
    def __init__(self):
        self.taps = []
        self._running = False

    def last_frame(self):
        return np.full((4, 4, 3), 7 if self._running else 9, np.uint8)

    def tap(self, x, y):
        self.taps.append((x, y))
        self._running = True


class OkMatcher:
    def find(self, frame, name, threshold=0.8):
        if name == "ok" and int(frame[0, 0, 0]) == 255:
            return (10, 10)
        return None

    def present(self, frame, name, threshold=0.8):
        return name == "slide" and int(frame[0, 0, 0]) == 7


class CloseMatcher:
    def find(self, frame, name, threshold=0.8):
        if name == "close" and int(frame[0, 0, 0]) == 9:
            return (30, 40)
        return None

    def present(self, frame, name, threshold=0.8):
        return name == "slide" and int(frame[0, 0, 0]) == 7


class BoostMatcher:
    def present(self, frame, name, threshold=0.8):
        state = int(frame[0, 0, 0])
        if state == 1:
            return name in {"pickboosts", "dblcheck"}
        if state == 2:
            return name == "dblbanner"
        return False

    def find(self, frame, name, threshold=0.8):
        if int(frame[0, 0, 0]) == 1 and name == "multibuy":
            return (20, 30)
        return None


class RequiredBoostMatcher:
    def __init__(self, active=True):
        self.active = active

    def present(self, frame, name, threshold=0.8):
        if name == "slide":
            return self.active
        if name == "multibtn":
            return not self.active
        if name == "tilecheck":
            return True
        return False

    def find(self, frame, name, threshold=0.8):
        if name in {"tile_hp", "tile_watch", "tile_x2", "play"}:
            return (100, 100)
        return None


def test_wait_for_result_frame_returns_first_ok_frame():
    pending = np.zeros((4, 4, 3), np.uint8)
    result = np.full((4, 4, 3), 255, np.uint8)
    clock = FakeClock()

    out = farm.wait_for_result_frame(
        FakeDevice([pending, result]),
        OkMatcher(),
        timeout_s=3.0,
        poll_s=0.2,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert out is result
    assert clock.t == 0.2


def test_wait_for_change_handles_reused_mutated_frame_object():
    clock = FakeClock()
    frame = np.zeros((20, 20, 3), np.uint8)

    class ReusedFrameDevice:
        def __init__(self):
            self.calls = 0

        def last_frame(self):
            self.calls += 1
            if self.calls >= 2:
                frame[:, :] = 255
            return frame

    farm._wait_for_change(
        ReusedFrameDevice(),
        frame,
        timeout_s=1.0,
        poll_s=0.2,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert clock.t == 0.2


def test_wait_for_result_frame_returns_none_without_ok():
    frame = np.zeros((4, 4, 3), np.uint8)
    clock = FakeClock()

    out = farm.wait_for_result_frame(
        FakeDevice([frame]),
        OkMatcher(),
        timeout_s=0.6,
        poll_s=0.2,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert out is None


def test_read_run_result_uses_result_frame(monkeypatch):
    result = np.full((4, 4, 3), 255, np.uint8)
    monkeypatch.setattr(
        farm_common,
        "read_results",
        lambda frame, cfg: {"coins": 123, "ingredients": 4},
    )

    out = farm.read_run_result(
        FakeDevice([result]),
        cfg=object(),
        matcher=OkMatcher(),
        timeout_s=1.0,
        sleep=lambda _: None,
        now=lambda: 0.0,
    )

    assert out == {"coins": 123, "ingredients": 4, "read_ok": True}


def test_read_run_result_does_not_read_without_result_ok(monkeypatch):
    frame = np.zeros((4, 4, 3), np.uint8)
    clock = FakeClock()
    monkeypatch.setattr(
        farm_common,
        "read_results",
        lambda frame, cfg: (_ for _ in ()).throw(AssertionError("must not read live frame")),
    )

    out = farm.read_run_result(
        FakeDevice([frame]),
        cfg=object(),
        matcher=OkMatcher(),
        timeout_s=0.6,
        poll_s=0.2,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert out == {"coins": 0, "ingredients": 0, "read_ok": False}


def test_read_run_result_waits_for_stable_value(monkeypatch):
    def result_frame(coins):
        frame = np.full((4, 4, 3), 255, np.uint8)
        frame[0, 0, 1] = coins
        return frame

    monkeypatch.setattr(
        farm_common,
        "read_results",
        lambda frame, cfg: {"coins": int(frame[0, 0, 1]), "ingredients": 0},
    )
    clock = FakeClock()

    out = farm.read_run_result(
        FakeDevice([result_frame(10), result_frame(20), result_frame(20), result_frame(20)]),
        cfg=object(),
        matcher=OkMatcher(),
        timeout_s=1.0,
        poll_s=0.5,
        settle_timeout_s=5.0,
        sleep=clock.sleep,
        now=clock.now,
    )

    assert out == {"coins": 20, "ingredients": 0, "read_ok": True}


def test_read_run_result_flags_settled_zero_as_unread(monkeypatch):
    """A genuine Result frame that reads 0 (OCR miss / animation) must be read_ok=False —
    a completed run never truly banks 0, so it must not be silently counted."""
    result = np.full((4, 4, 3), 255, np.uint8)
    monkeypatch.setattr(
        farm_common, "read_results",
        lambda frame, cfg: {"coins": 0, "ingredients": 0},
    )
    out = farm.read_run_result(
        FakeDevice([result]), cfg=object(), matcher=OkMatcher(),
        timeout_s=1.0, poll_s=0.5, settle_timeout_s=1.0,
        sleep=lambda _: None, now=lambda: 0.0,
    )
    assert out == {"coins": 0, "ingredients": 0, "read_ok": False}


def test_read_run_result_flags_zero_coins_with_ingredients_as_unread(monkeypatch):
    """A coin OCR miss to 0 while ingredients read fine must be read_ok=False — a completed
    run never truly banks 0 coins, so the coin read is UNREAD even if ingredients came through."""
    result = np.full((4, 4, 3), 255, np.uint8)
    monkeypatch.setattr(
        farm_common, "read_results",
        lambda frame, cfg: {"coins": 0, "ingredients": 7},
    )
    out = farm.read_run_result(
        FakeDevice([result]), cfg=object(), matcher=OkMatcher(),
        timeout_s=1.0, poll_s=0.5, settle_timeout_s=1.0,
        sleep=lambda _: None, now=lambda: 0.0,
    )
    assert out == {"coins": 0, "ingredients": 7, "read_ok": False}


class MenuWalletMatcher:
    def __init__(self, play=True):
        self._play = play

    def find(self, frame, name, threshold=0.8):
        return (5, 5) if (name == "play" and self._play) else None

    def present(self, frame, name, threshold=0.8):
        return False


def test_read_wallet_reads_when_menu_visible(monkeypatch):
    class Cfg:
        regions = {"coin_counter": object()}
        templates_dir = "t"
    monkeypatch.setattr(farm_common, "read_int", lambda f, region, td: 2945456)
    out = farm.read_wallet(
        FakeDevice([np.full((4, 4, 3), 255, np.uint8)]),
        Cfg(), MenuWalletMatcher(play=True), tries=2, sleep=lambda _: None)
    assert out == 2945456


def test_read_wallet_refuses_off_menu(monkeypatch):
    class Cfg:
        regions = {"coin_counter": object()}
        templates_dir = "t"
    read_calls = []
    monkeypatch.setattr(
        farm_common, "read_int",
        lambda *a: (read_calls.append(1), 999)[1])
    out = farm.read_wallet(
        FakeDevice([np.zeros((4, 4, 3), np.uint8)]),
        Cfg(), MenuWalletMatcher(play=False), tries=2, sleep=lambda _: None)
    assert out is None
    assert not read_calls           # never OCRs a non-menu frame


def test_ensure_running_taps_visible_ok_on_animated_result(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    dev = AnimatedOkDevice()

    assert farm.ensure_running(dev, OkMatcher(), tries=8, log=lambda _: None) is True
    assert dev.taps == [(10, 10)]


def test_ensure_running_closes_stray_popup(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    dev = PopupDevice()

    assert farm.ensure_running(dev, CloseMatcher(), tries=4, log=lambda _: None) is True
    assert dev.taps == [(30, 40)]


def test_ensure_running_drains_gifts_before_menu_navigation(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    calls = []

    def fake_draw_gifts(*args, **kwargs):
        calls.append("draw_gifts")
        return GiftDrawResult(draws=2, depleted=True, opened=True)

    class GiftOnlyMatcher:
        def find(self, frame, name, threshold=0.8):
            if name == "giftbtn":
                return (1, 4)
            if name == "play":
                return (3, 3)
            return None

        def present(self, frame, name, threshold=0.8):
            return False

    monkeypatch.setattr(farm, "draw_gifts", fake_draw_gifts)
    gift_state = {}

    assert farm.ensure_running(
        FakeDevice([np.zeros((4, 4, 3), np.uint8)]),
        GiftOnlyMatcher(),
        tries=2,
        log=lambda _: None,
        gift_state=gift_state,
    ) is False

    assert calls == ["draw_gifts"]
    assert gift_state["depleted"] is True


def test_restart_game_does_not_wait_full_splash(monkeypatch):
    calls = []
    sleeps = []

    class Cfg:
        adb_path = "adb"
        device_serial = "127.0.0.1:5555"

    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: calls.append(cmd))
    monkeypatch.setattr(
        farm,
        "_sleep_interruptible",
        lambda seconds, should_stop=None: sleeps.append(seconds),
    )

    farm._restart_game(Cfg(), log=lambda _: None)

    assert calls[0][-3:] == ["am", "force-stop", "com.devsisters.crg"]
    assert calls[1][-2] == "-n"
    assert calls[1][-1] == "com.devsisters.crg/com.devsisters.CookieRunForKakao.OvenbreakX"
    assert sleeps == [1.5, 3.0]


def test_play_until_death_never_actions_without_slide_hud(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_remaining", lambda *a, **k: None)

    class Agent:
        def reset(self):
            pass

        def decide(self, frame):
            return farm.ActionDecision(1, "test")

    class NoSlideMatcher:
        def present(self, frame, name, threshold=0.8):
            return False

        def find(self, frame, name, threshold=0.8):
            return None

    class Cfg:
        decision_hz = 60
        gestures = type("G", (), {"jump_button": (1, 2), "slide_button": (3, 4),
                                  "slide_hold_ms": 300})()

    dev = FakeDevice([np.zeros((4, 4, 3), np.uint8)])
    dev.taps = []
    dev.holds = []
    dev.tap = lambda x, y: dev.taps.append((x, y))
    dev.hold = lambda x, y, duration_ms: dev.holds.append((x, y, duration_ms))

    farm.play_until_death(dev, Cfg(), Agent(), NoSlideMatcher(), max_s=1.0, log=lambda _: None)

    assert dev.taps == []
    assert dev.holds == []


def test_auto_serial_config_switches_stale_single_device(monkeypatch):
    @dataclass(frozen=True)
    class Cfg:
        adb_path: str = ""
        device_serial: str = "emulator-5554"

    logs = []
    monkeypatch.setattr(farm, "_ready_adb_devices", lambda adb_path="": ["127.0.0.1:5555"])

    cfg = farm._auto_serial_config(Cfg(), log=logs.append)

    assert cfg.device_serial == "127.0.0.1:5555"
    assert logs == ["[adb] using 127.0.0.1:5555 instead of emulator-5554"]


def test_auto_serial_config_keeps_ambiguous_missing_device(monkeypatch):
    @dataclass(frozen=True)
    class Cfg:
        adb_path: str = ""
        device_serial: str = "missing"

    monkeypatch.setattr(farm, "_ready_adb_devices", lambda adb_path="": ["a", "b"])

    cfg = farm._auto_serial_config(Cfg(), log=lambda _: None)

    assert cfg is not None
    assert cfg.device_serial == "missing"


def test_buy_double_coins_tracks_spend_before_success(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    dialog = np.full((4, 4, 3), 1, np.uint8)
    banner = np.full((4, 4, 3), 2, np.uint8)
    dev = FakeDevice([dialog, banner])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.buy_double_coins(
        dev,
        BoostMatcher(),
        SpendingConfig(
            allow_coin_boosts=True,
            max_boost_cost_per_run=1200,
            max_double_coin_rolls=2,
        ),
        log=lambda _: None,
    )

    assert result.active is True
    assert result.spent == 1200
    assert dev.taps == [(20, 30)]


def test_buy_double_coins_does_not_tap_when_budget_too_low(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    dialog = np.full((4, 4, 3), 1, np.uint8)
    dev = FakeDevice([dialog])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.buy_double_coins(
        dev,
        BoostMatcher(),
        SpendingConfig(
            allow_coin_boosts=True,
            max_boost_cost_per_run=500,
            max_double_coin_rolls=1,
        ),
        log=lambda _: None,
    )

    assert result.active is False
    assert result.spent == 0
    assert dev.taps == []


def test_ensure_running_does_not_play_without_double_coin_banner(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    monkeypatch.setattr(farm, "ensure_run_boosts", lambda *a, **k: farm.BoostResult(True, 1600))
    monkeypatch.setattr(farm, "buy_double_coins", lambda *a, **k: farm.BoostResult(False, 1200))
    play_taps = []
    monkeypatch.setattr(
        farm,
        "_tap_template",
        lambda dev, matcher, name, thresh=0.72: play_taps.append(name) or True,
    )
    dev = FakeDevice([np.zeros((400, 500, 3), np.uint8)])   # big enough for the tile-badge ROI at pt
    logs = []
    cycle = {}

    assert farm.ensure_running(
        dev,
        RequiredBoostMatcher(active=False),
        cfg=type("Cfg", (), {"spending": _SPEND})(),
        tries=2,
        log=logs.append,
        cycle=cycle,
    ) is False

    assert "play" not in play_taps
    assert logs[0].startswith("[boost] ready=False required=True double_banner=False")
    assert logs[1] == "[boost] Double Coins banner not verified; not pressing Play"
    assert cycle["required_boost_cost"] == 1600
    assert cycle["double_coin_cost"] == 1200
    assert cycle["boost_cost"] == 0
    assert cycle["double_coin_failed"] is True


def test_ensure_running_does_not_play_without_required_three_boosts(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    monkeypatch.setattr(farm, "ensure_run_boosts", lambda *a, **k: farm.BoostResult(False, 800))
    play_taps = []
    monkeypatch.setattr(
        farm,
        "_tap_template",
        lambda dev, matcher, name, thresh=0.72: play_taps.append(name) or True,
    )
    dev = FakeDevice([np.zeros((400, 500, 3), np.uint8)])   # big enough for the tile-badge ROI at pt
    logs = []
    cycle = {}

    assert farm.ensure_running(
        dev,
        RequiredBoostMatcher(active=False),
        cfg=type("Cfg", (), {"spending": _SPEND})(),
        tries=2,
        log=logs.append,
        cycle=cycle,
    ) is False

    assert "play" not in play_taps
    assert logs[0].startswith("[boost] ready=False required=True double_banner=False")
    assert logs[1] == "[boost] required three boost tiles not verified; not pressing Play"
    assert cycle["boost_cost"] == 0
    assert cycle["required_boost_cost"] == 0
    assert cycle["double_coin_cost"] == 0


def test_ensure_running_does_not_retry_double_coin_after_failed_verification(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    monkeypatch.setattr(farm, "ensure_run_boosts", lambda *a, **k: farm.BoostResult(True, 1600))
    calls = []

    def fail_double(*args, **kwargs):
        calls.append(1)
        return farm.BoostResult(False, 1200)

    monkeypatch.setattr(farm, "buy_double_coins", fail_double)
    monkeypatch.setattr(farm, "_tap_template", lambda *a, **k: True)
    dev = FakeDevice([np.zeros((4, 4, 3), np.uint8)])
    logs = []
    cycle = {}

    assert farm.ensure_running(
        dev,
        RequiredBoostMatcher(active=False),
        cfg=type("Cfg", (), {"spending": _SPEND})(),
        tries=4,
        log=logs.append,
        cycle=cycle,
    ) is False

    assert len(calls) == 1
    assert cycle["boost_cost"] == 0
    assert logs[-1] == "[boost] Double Coins was already attempted and not verified; not retrying"


def test_ensure_running_books_boost_cost_only_when_play_gate_ready(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    monkeypatch.setattr(farm, "ensure_run_boosts", lambda *a, **k: farm.BoostResult(True, 1600))
    monkeypatch.setattr(farm, "buy_double_coins", lambda *a, **k: farm.BoostResult(True, 1200))
    tapped = []
    monkeypatch.setattr(
        farm,
        "_tap_template",
        lambda dev, matcher, name, thresh=0.72: tapped.append(name) or True,
    )
    dev = FakeDevice([np.zeros((4, 4, 3), np.uint8)])
    cycle = {}

    assert farm.ensure_running(
        dev,
        RequiredBoostMatcher(active=False),
        cfg=type("Cfg", (), {"spending": _SPEND})(),
        tries=2,
        log=lambda _: None,
        cycle=cycle,
    ) is False

    assert "play" in tapped
    assert cycle["boost_cost"] == 2800


class TileMatcher:
    """Tile check state encoded in frame[0,0,0]: 0 = tiles unchecked, 1 = checked."""

    def find(self, frame, name, threshold=0.8):
        if name in {"tile_hp", "tile_watch", "tile_x2"}:
            return (100, 100)
        return None

    def present(self, roi, name, threshold=0.8):
        return name == "tilecheck" and int(roi[0, 0, 0]) == 1


_SPEND = SpendingConfig(allow_coin_boosts=True, max_boost_cost_per_run=12000)


def test_ensure_run_boosts_counts_checked_tiles_without_tapping(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    checked = np.full((400, 400, 3), 1, np.uint8)
    dev = FakeDevice([checked])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.ensure_run_boosts(dev, TileMatcher(), _SPEND, log=lambda _: None)

    assert result == farm.BoostResult(True, 1600)  # HP + watch + x2 star (stock, 0)
    assert dev.taps == []                   # never tap a checked tile (it would toggle OFF)


def test_ensure_run_boosts_taps_unchecked_tile_once_then_verifies(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    unchecked = np.full((400, 400, 3), 0, np.uint8)
    checked = np.full((400, 400, 3), 1, np.uint8)
    # badge-first fast read + the sharp _find_stable read both see the CURRENT (unchecked) screen,
    # then the post-tap re-verify sees checked (two unchecked frames: one per read before the tap).
    dev = FakeDevice([unchecked, unchecked, checked])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.ensure_run_boosts(dev, TileMatcher(), _SPEND, log=lambda _: None)

    assert result == farm.BoostResult(True, 1600)
    assert dev.taps == [(100, 100)]         # exactly one enable tap, at the tile centre


def test_ensure_run_boosts_skips_invisible_tiles(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)

    class NoTiles:
        def find(self, frame, name, threshold=0.8):
            return None

        def present(self, roi, name, threshold=0.8):
            return False

    dev = FakeDevice([np.zeros((400, 400, 3), np.uint8)])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.ensure_run_boosts(dev, NoTiles(), _SPEND, log=lambda _: None)

    assert result == farm.BoostResult(False, 0)
    assert dev.taps == []


class X2DepletedMatcher:
    """hp & watch verify checked; the x2 tile is visible but never checks (owned stock
    depleted). Records the last tile find() was asked about so tilecheck can answer per-tile."""

    def __init__(self):
        self._last = None

    def find(self, frame, name, threshold=0.8):
        if name in {"tile_hp", "tile_watch", "tile_x2"}:
            self._last = name
            return (100, 100)
        return None

    def present(self, roi, name, threshold=0.8):
        return name == "tilecheck" and self._last in {"tile_hp", "tile_watch"}


def test_ensure_run_boosts_treats_x2_stock_depletion_as_best_effort(monkeypatch):
    monkeypatch.setattr(farm, "_sleep_interruptible", lambda *a, **k: None)
    monkeypatch.setattr(farm, "_wait_for_change", lambda *a, **k: None)
    dev = FakeDevice([np.zeros((400, 400, 3), np.uint8)])
    dev.taps = []
    dev.tap = lambda x, y: dev.taps.append((x, y))

    result = farm.ensure_run_boosts(dev, X2DepletedMatcher(), _SPEND, log=lambda _: None)

    # HP + watch verified (1600); x2 can't be checked (depleted stock) but is best-effort,
    # so the gate must still report active rather than halt the farm.
    assert result == farm.BoostResult(True, 1600)
