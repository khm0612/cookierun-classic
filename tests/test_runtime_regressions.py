from __future__ import annotations

import ast
import builtins
import importlib
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from cookierun_bot import farm_cards


ROOT = Path(__file__).resolve().parents[1]


def _load_defs(path: Path, *names: str, globals_: dict | None = None) -> dict:
    """Load selected script definitions without running hardware-oriented module setup."""
    # ponytail: legacy scripts act at import; selecting pure defs avoids a broad entrypoint refactor.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace = dict(globals_ or {})
    namespace.setdefault("__builtins__", __builtins__)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def monitor(monkeypatch):
    module = importlib.import_module("monitor")
    module._STOP.clear()
    module._sup.update(target=0, done=0, finished=False, proc=None)
    monkeypatch.setattr(module, "_CARD_FLAG", str(ROOT / "data" / "_test_card_active"))
    yield module
    module._STOP.clear()
    module._sup.update(target=0, done=0, finished=False, proc=None)


def test_farm_card_handler_never_taps_even_when_legacy_auto_env_is_set(monkeypatch):
    class Device:
        taps = []

        def last_frame(self):
            return np.zeros((1440, 2560, 3), dtype=np.uint8)

        def tap(self, x, y):
            self.taps.append((x, y))

    class Matcher:
        def present(self, frame, name, threshold):
            return True

    calls = 0

    def should_stop():
        nonlocal calls
        calls += 1
        return calls > 1

    monkeypatch.setenv("AIFARM_CARD_AUTO", "1")
    monkeypatch.setattr(farm_cards, "_alert_user", lambda: None)
    monkeypatch.setattr(farm_cards, "_sleep_interruptible", lambda *a, **k: None)

    dev = Device()
    farm_cards._cardgame(dev, Matcher(), should_stop=should_stop)

    assert dev.taps == []


def test_manual_card_wait_exits_without_capture_when_batch_finished(monitor, monkeypatch):
    monitor._sup["finished"] = True
    monkeypatch.setattr(monitor, "grab", lambda: pytest.fail("finished wait grabbed a frame"))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: pytest.fail("finished wait slept"))

    monitor._wait_for_manual_card_clear(object())


def test_capture_failure_does_not_clear_card_protection(monitor, monkeypatch):
    class StopLoop(Exception):
        pass

    cleared = []
    monkeypatch.setattr(monitor, "grab", lambda: None)
    monkeypatch.setattr(monitor, "_set_card_flag", lambda: None)
    monkeypatch.setattr(monitor, "_clear_card_flag", lambda: cleared.append(True))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: (_ for _ in ()).throw(StopLoop()))

    with pytest.raises(StopLoop):
        monitor._wait_for_manual_card_clear(object())

    assert cleared == []


def test_manual_card_wait_recovers_repeated_capture_failures(monitor, monkeypatch):
    reconnects = []
    grabs = []
    monkeypatch.setattr(monitor, "grab", lambda: grabs.append(True) or None)
    monkeypatch.setattr(monitor, "_set_card_flag", lambda: None)
    monkeypatch.setattr(monitor, "_clear_card_flag", lambda: None)
    monkeypatch.setattr(monitor, "_alert_user", lambda: None)
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)

    def reconnect():
        reconnects.append(True)
        monitor._sup["finished"] = True

    monkeypatch.setattr(monitor, "reconnect_adb", reconnect)

    monitor._wait_for_manual_card_clear(object())

    assert len(grabs) == monitor.GRAB_FAILS_BEFORE_RECONNECT
    assert reconnects == [True]


def test_repeated_solver_capture_failures_leave_card_protection_armed(monitor, monkeypatch):
    cleared = []
    monkeypatch.setattr(monitor, "grab", lambda: None)
    monkeypatch.setattr(monitor, "_clear_card_flag", lambda: cleared.append(True))
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)

    monitor.solve_cardgame(object())

    assert cleared == []


def test_emulator_refresh_waits_until_card_solver_releases_ownership(monitor, monkeypatch):
    solver_entered = threading.Event()
    solver_release = threading.Event()
    refreshed = threading.Event()

    def fake_solver(_matcher):
        solver_entered.set()
        assert solver_release.wait(2)

    monkeypatch.setattr(monitor, "_solve_cardgame", fake_solver)
    monkeypatch.setattr(monitor, "refresh_emulator", lambda emit: refreshed.set() or True)

    solver = threading.Thread(target=monitor.solve_cardgame, args=(object(),))
    solver.start()
    assert solver_entered.wait(1)

    refresh = threading.Thread(target=monitor._refresh_emulator_safely, args=(lambda _: None,))
    refresh.start()
    time.sleep(0.05)
    assert not refreshed.is_set()

    solver_release.set()
    solver.join(1)
    refresh.join(1)
    assert refreshed.is_set()


def test_stray_cleanup_is_scoped_to_this_worktree(monitor, monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    monitor._kill_stray_farm()

    command = calls[0][0][0][-1]
    assert monitor._SESSION_ID in command
    assert str(monitor.ROOT).replace("\\", "\\\\") not in command
    assert "ai_farm\\.py|supervisor\\.py" not in command


def test_cleanup_session_marker_is_stable_scope_hash(monitor):
    assert len(monitor._SESSION_ID) == 16
    int(monitor._SESSION_ID, 16)
    expected = monitor.hashlib.sha256(monitor.SERIAL.casefold().encode()).hexdigest()[:16]
    assert monitor._SESSION_ID == expected


def test_device_lock_refuses_a_second_monitor_owner(monitor, tmp_path):
    path = tmp_path / "monitor.lock"
    first = monitor._acquire_device_lock(path)
    assert first is not None
    try:
        assert monitor._acquire_device_lock(path) is None
    finally:
        first.close()
    third = monitor._acquire_device_lock(path)
    assert third is not None
    third.close()


def test_supervisor_is_not_launched_after_shutdown_wins(monitor, monkeypatch):
    launches = []
    monitor._STOP.clear()
    monitor._sup.update(target=1, done=0, finished=False, proc=None)
    monkeypatch.setattr(monitor, "_kill_stray_farm", monitor._STOP.set)
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: launches.append(True))
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(OSError()))

    monitor._pump_supervisor(1)

    assert launches == []


def test_modal_taps_stand_down_during_emulator_refresh(monitor, monkeypatch):
    monitor._REFRESH_PENDING.set()
    monkeypatch.setattr(
        monitor,
        "dismiss_modal",
        lambda *a, **k: pytest.fail("modal tapped during refresh"),
    )
    try:
        assert monitor._dismiss_modal_safely(object(), "league", (1, 2), "league") is False
    finally:
        monitor._REFRESH_PENDING.clear()

    class Matcher:
        def present(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(monitor, "grab", lambda: np.zeros((2, 2, 3), dtype=np.uint8))
    assert monitor._dismiss_modal_safely(Matcher(), "league", (1, 2), "league") is False


def test_monitor_returns_failure_when_supervision_is_incomplete(monitor, monkeypatch):
    class Matcher:
        def __init__(self, _):
            pass

        def has(self, _):
            return True

    class Thread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            monitor._sup.update(target=3, done=2, finished=True, proc=None)

    monkeypatch.setattr(monitor, "TemplateMatcher", Matcher)
    monkeypatch.setattr(monitor.threading, "Thread", Thread)
    monkeypatch.setattr(monitor, "_kill_stray_farm", lambda: None)

    assert monitor.main(supervise_target=3) == 1


def test_supervisor_returns_failure_when_target_is_incomplete(monkeypatch):
    supervisor = importlib.import_module("supervisor")

    class Child:
        stdout = iter(())

        def wait(self):
            return 2

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)

    assert supervisor.main(1) != 0


def test_supervisor_propagates_session_marker_to_farm_child(monkeypatch):
    supervisor = importlib.import_module("supervisor")
    commands = []

    class Child:
        stdout = iter((">> RESULT: ok\n",))

        def wait(self):
            return 0

    monkeypatch.setattr(supervisor, "SESSION_ARG", "--cookierun-session=test-123")
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or Child(),
    )

    assert supervisor.main(1) == 0
    assert commands[0][-1] == "--cookierun-session=test-123"


def test_pit_labels_require_explicit_recorded_pit_evidence():
    ns = _load_defs(ROOT / "scripts" / "mine_negatives.py", "_pit_times")
    pit_times = ns["_pit_times"]

    assert pit_times({"duration_s": 20.0, "frames": [{"t": 20.0}]}) == []
    assert pit_times({"duration_s": 20.0, "pit_times": [12.5, 25, "bad"]}) == [12.5]
    assert pit_times({"complete": False, "duration_s": 20.0, "frames": [{"t": 20.0}],
                      "pit_times": [12.5]}) == []
    assert pit_times({"duration_s": 20.0, "frames": [{"t": 8.0}],
                      "pit_times": [12.5]}) == []


def test_iql_rejects_explicitly_incomplete_recordings():
    recording_is_complete = importlib.import_module("_runtime").recording_is_complete

    assert recording_is_complete({"frames": [{"idx": 0}]}) is True
    assert recording_is_complete({"complete": True, "frames": [{"idx": 0}]}) is True
    assert recording_is_complete({"complete": False, "frames": [{"idx": 0}]}) is False
    assert recording_is_complete({"complete": True, "frames": []}) is False


def test_recording_writer_does_not_deadlock_or_claim_failed_writes(tmp_path, monkeypatch):
    ns = _load_defs(
        ROOT / "scripts" / "ai_farm.py",
        "_RecordingWriter",
        globals_={"os": __import__("os"), "queue": queue, "threading": threading, "time": time},
    )
    writer_type = ns["_RecordingWriter"]

    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    writer = writer_type(tmp_path, maxsize=1)
    writer.submit(0, 0.1, b"jpeg")

    started = time.monotonic()
    writer.close(timeout=0.5)

    assert time.monotonic() - started < 1.0
    assert writer.frames == []
    assert isinstance(writer.error, OSError)


def test_recording_writer_freezes_metadata_after_close_timeout(tmp_path, monkeypatch):
    ns = _load_defs(
        ROOT / "scripts" / "ai_farm.py",
        "_RecordingWriter",
        globals_={"os": __import__("os"), "queue": queue, "threading": threading, "time": time},
    )
    writer_type = ns["_RecordingWriter"]
    entered = threading.Event()
    release = threading.Event()

    class SlowFile:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def write(self, payload):
            entered.set()
            assert release.wait(2)
            return len(payload)

    monkeypatch.setattr(builtins, "open", lambda *a, **k: SlowFile())
    writer = writer_type(tmp_path)
    writer.submit(0, 0.1, b"jpeg")
    assert entered.wait(1)

    assert writer.close(timeout=0.05) is False
    assert writer.frames == []
    assert isinstance(writer.error, TimeoutError)

    release.set()
    writer.thread.join(1)
    assert writer.frames == []


def test_diagnostic_names_include_a_process_session():
    ns = _load_defs(
        ROOT / "scripts" / "ai_farm.py",
        "_diag_stem",
        globals_={"_DIAG_SESSION": "123"},
    )

    first = ns["_diag_stem"](1, "123")
    second = ns["_diag_stem"](1, "456")
    assert first != second
    assert first.startswith("r") and first[1:].isdigit()


def test_ai_farm_has_no_independent_headstart_tap_loop():
    source = (ROOT / "scripts" / "ai_farm.py").read_text(encoding="utf-8-sig")

    assert "Head Start settled" not in source


def test_self_farm_recorder_only_lists_successfully_written_frames(tmp_path, monkeypatch):
    ns = _load_defs(
        ROOT / "scripts" / "self_farm.py",
        "make_recorder",
        globals_={
            "os": __import__("os"),
            "queue": queue,
            "threading": threading,
            "time": time,
            "cv2": importlib.import_module("cv2"),
            "_pitfall": lambda _frame: False,
            "SAVE_FPS": 35,
            "SAVE_W": 64,
            "ACTION_NOOP": 0,
            "ACTION_JUMP": 1,
            "ACTION_SLIDE": 2,
        },
    )
    monkeypatch.setattr(ns["cv2"], "imwrite", lambda *a, **k: False)
    on_step, frames, _keys, close = ns["make_recorder"](str(tmp_path))
    decision = type("Decision", (), {"action": 0})()

    on_step(1.0, np.zeros((64, 64, 3), dtype=np.uint8), decision)
    closed, _pit_times, error = close()

    assert frames == []
    assert closed is True
    assert isinstance(error, OSError)


def test_self_farm_recorder_freezes_metadata_and_closes_terminal_slide(tmp_path, monkeypatch):
    ns = _load_defs(
        ROOT / "scripts" / "self_farm.py",
        "make_recorder",
        globals_={
            "os": __import__("os"),
            "queue": queue,
            "threading": threading,
            "time": time,
            "cv2": importlib.import_module("cv2"),
            "_pitfall": lambda _frame: False,
            "SAVE_FPS": 35,
            "SAVE_W": 64,
            "ACTION_NOOP": 0,
            "ACTION_JUMP": 1,
            "ACTION_SLIDE": 2,
        },
    )
    entered = threading.Event()
    release = threading.Event()

    def slow_write(*_args, **_kwargs):
        entered.set()
        assert release.wait(2)
        return True

    monkeypatch.setattr(ns["cv2"], "imwrite", slow_write)
    on_step, frames, keys, close = ns["make_recorder"](str(tmp_path))
    slide = type("Decision", (), {"action": 2})()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    on_step(1.0, frame, slide)
    on_step(2.0, frame, slide)
    assert entered.wait(1)

    closed, _pit_times, error = close(timeout=0.05)
    assert closed is False
    assert error is None
    assert frames == []
    assert keys == [{"t": 1.0, "action": "slide", "dur": 1.0}]

    release.set()
    time.sleep(0.05)
    assert frames == []


def test_self_farm_rejects_incomplete_recording_and_dead_monitor(tmp_path):
    class Log:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class DeadMonitor:
        def wait(self, timeout):
            return 1

    fake_subprocess = type(
        "Subprocess",
        (),
        {
            "STDOUT": object(),
            "TimeoutExpired": TimeoutError,
            "Popen": staticmethod(lambda *a, **k: DeadMonitor()),
        },
    )
    ns = _load_defs(
        ROOT / "scripts" / "self_farm.py",
        "_launch_monitor",
        "_recording_usable",
        globals_={
            "subprocess": fake_subprocess,
            "open": lambda *a, **k: Log(),
            "PY": "python",
            "ROOT": tmp_path,
            "MIN_DUR_S": 5.0,
            "MIN_FRAMES": 100,
        },
    )

    assert ns["_launch_monitor"]() is None
    assert ns["_recording_usable"](False, 60.0, 1000) is False
    assert ns["_recording_usable"](True, 60.0, 1000) is True


def test_pit_negative_uses_prompt_lag_and_default_finds_bot_runs(tmp_path):
    ns = _load_defs(
        ROOT / "scripts" / "mine_negatives.py",
        "_decision_time",
        "_default_run_names",
        globals_={
            "os": __import__("os"),
            "glob": importlib.import_module("glob"),
            "PRE_S": 0.30,
            "PIT_PROMPT_LAG_S": 0.75,
        },
    )
    for name in ("demo_self_one", "botrun_two"):
        run = tmp_path / name
        run.mkdir()
        (run / "frames.json").write_text("{}", encoding="utf-8")

    assert ns["_decision_time"](10.0, "hit") == pytest.approx(9.70)
    assert ns["_decision_time"](10.0, "pit") == pytest.approx(8.95)
    assert ns["_default_run_names"](str(tmp_path)) == ["botrun_two", "demo_self_one"]


@pytest.mark.parametrize(
    "raw",
    ["0.4", "0.4,", "fast,0.6", "0.4,0.5,0.6", "nan,0.5", "inf,0.5", "-0.1,0.5", "1.1,0.5"],
)
def test_hybrid_conf_parser_rejects_invalid_values_with_clear_error(raw):
    ns = _load_defs(ROOT / "scripts" / "ai_farm.py", "_parse_hybrid_confs")

    with pytest.raises(SystemExit, match="AIFARM_HYBRID_CONFS.*two numbers"):
        ns["_parse_hybrid_confs"](raw, 0.6)


def test_hybrid_conf_parser_accepts_exactly_two_numbers():
    ns = _load_defs(ROOT / "scripts" / "ai_farm.py", "_parse_hybrid_confs")

    assert ns["_parse_hybrid_confs"]("0.4,0.7", 0.6) == (0.4, 0.7)
    assert ns["_parse_hybrid_confs"]("", 0.6) == (0.6, 0.6)
