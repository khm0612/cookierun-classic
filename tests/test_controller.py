import pytest
from types import SimpleNamespace

from cookierun_bot.agents.controller import (
    adb_ready_devices,
    controller_runtime_config,
    format_action_brief,
    format_action_line,
    format_boost_line,
    format_coin_line,
    format_rate_line,
    format_status_line,
    parse_adb_devices,
    parse_max_runs,
    summarize_action_log,
    summarize_boost_log,
)
from cookierun_bot.agents.action_watch import ActionWatchSample
from cookierun_bot.config import Config, ConfigError, Gestures, Region, RewardWeights
from cookierun_bot.device import select_adb_serial


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, 60, "Episode 1",
                  {"play_area": r, "coin_counter": r, "mystery_box_counter": r,
                   "results_coins": r, "results_ingredients": r},
                  Gestures((1, 2), (3, 4), 300), RewardWeights(1, 50, 0.01, 10),
                  ["start"], ["buy"], "templates")


def test_parse_max_runs_allows_blank_for_unlimited():
    assert parse_max_runs("") is None
    assert parse_max_runs("  ") is None
    assert parse_max_runs("0") is None


def test_parse_max_runs_requires_positive_integer():
    assert parse_max_runs("3") == 3
    with pytest.raises(ValueError):
        parse_max_runs("-1")


def test_parse_max_runs_rejects_text():
    with pytest.raises(ValueError, match="number"):
        parse_max_runs("many")


def test_parse_adb_devices_only_returns_ready_devices():
    out = """List of devices attached
127.0.0.1:5555\tdevice
emulator-5554\toffline
ABC123\tunauthorized

"""
    assert parse_adb_devices(out) == ["127.0.0.1:5555"]


def test_adb_ready_devices_parses_success(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout="List of devices attached\n127.0.0.1:5555\tdevice\n",
        ),
    )

    assert adb_ready_devices("adb") == ["127.0.0.1:5555"]


def test_adb_ready_devices_returns_empty_on_adb_error(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )

    assert adb_ready_devices("adb") == []


def test_select_adb_serial_auto_switches_only_single_ready_device():
    assert select_adb_serial("", ["emulator-5554"]) == ("emulator-5554", "ready")
    assert select_adb_serial("127.0.0.1:5555", ["emulator-5554"]) == (
        "emulator-5554",
        "ready",
    )
    assert select_adb_serial("missing", ["a", "b"]) == ("missing", "device missing")
    assert select_adb_serial("", []) == ("", "no devices")


def test_status_line_formatting():
    assert (
        format_status_line("running", "3", "127.0.0.1:5555")
        == "Status: running    Runs: 3    Device: 127.0.0.1:5555"
    )
    assert format_status_line("idle", "0", "") == "Status: idle    Runs: 0    Device: no device"
    assert format_coin_line("100", "250", "150") == "Latest coins: 100    Total: 250    Net: 150"
    assert (
        format_coin_line("108452", "34387997", "34279545")
        == "Latest coins: 108,452    Total: 34,387,997    Net: 34,279,545"
    )
    assert format_rate_line("500/hr", "1.2/hr") == "Net/hr: 500/hr    Ingredients/hr: 1.2/hr"
    assert format_rate_line("15062/hr", "0.0/hr") == "Net/hr: 15,062/hr    Ingredients/hr: 0.0/hr"
    assert format_boost_line("") == "Boost gate: not checked"
    assert format_boost_line("ready") == "Boost gate: ready"
    assert format_action_line("") == "Last action: none"
    assert format_action_line("jump reason=pit confirmed=1") == (
        "Last action: jump reason=pit confirmed=1"
    )


def test_action_brief_marks_read_only_advisor_output():
    sample = ActionWatchSample(
        1, 0.0, 1, "jump", "pit", 1, True, None, False, False, False, "[frame 1]"
    )

    assert format_action_brief(sample) == "jump reason=pit confirmed=1"


def test_log_summary_helpers():
    assert summarize_action_log("[action] jump reason=pit confirmed=1") == (
        "jump reason=pit confirmed=1"
    )
    assert summarize_action_log("[boost] ready=True") is None
    assert summarize_boost_log(
        "[boost] ready=False required=True double_banner=False tile_hp=checked"
    ) == "ready=False required=True double=False"
    assert summarize_boost_log(
        "[boost] Double Coins banner not verified; not pressing Play"
    ) == "Double Coins banner not verified; not pressing Play"
    assert summarize_boost_log("[action] jump reason=pit confirmed=1") is None


def test_controller_runtime_config_applies_ui_overrides():
    cfg = controller_runtime_config(_cfg(), "127.0.0.1:5555", "adb", "C:/adb.exe")

    assert cfg.device_serial == "127.0.0.1:5555"
    assert cfg.capture_backend == "adb"
    assert cfg.adb_path == "C:/adb.exe"


def test_controller_runtime_config_rejects_unknown_capture():
    with pytest.raises(ConfigError):
        controller_runtime_config(_cfg(), "", "bad", "")
