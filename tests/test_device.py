import threading

import numpy as np
import pytest

from cookierun_bot.device import (
    LDPlayerDevice,
    ScrcpyDevice,
    resolve_adb_serial,
    scrcpy_server_jar_path,
)


class _StoppedResource:
    def __init__(self):
        self.stopped = False
        self.released = False

    def stop(self):
        self.stopped = True

    def release(self):
        self.released = True


def test_scrcpy_server_jar_path_uses_package_location(monkeypatch, tmp_path):
    pkg = tmp_path / "scrcpy"
    pkg.mkdir()
    jar = pkg / "server.jar"
    jar.write_bytes(b"jar")

    class Spec:
        submodule_search_locations = [str(pkg)]

    monkeypatch.setattr("importlib.util.find_spec", lambda name: Spec if name == "scrcpy" else None)

    assert scrcpy_server_jar_path("server.jar") == str(jar)


def test_resolve_adb_serial_does_not_switch_an_explicit_missing_device(monkeypatch):
    monkeypatch.setattr("cookierun_bot.device.ready_adb_serials", lambda: ["127.0.0.1:5555"])

    assert resolve_adb_serial("emulator-5554") == "emulator-5554"


def test_resolve_adb_serial_keeps_requested_when_ambiguous(monkeypatch):
    monkeypatch.setattr("cookierun_bot.device.ready_adb_serials", lambda: ["a", "b"])

    assert resolve_adb_serial("missing") == "missing"


def test_scrcpy_stop_does_not_leave_a_stale_frame():
    device = ScrcpyDevice()
    device._latest = np.ones((2, 3, 3), dtype=np.uint8)

    device.stop()

    assert device.last_frame() is None


def test_scrcpy_start_rejects_an_overlapping_decoder_thread():
    device = ScrcpyDevice()
    device._thread = type("LiveThread", (), {"is_alive": lambda self: True})()

    with pytest.raises(RuntimeError, match="decoder thread is still running"):
        device.start()


def test_scrcpy_stop_retains_a_decoder_thread_that_did_not_exit():
    class SlowThread:
        joined = False

        def join(self, timeout):
            self.joined = True

        def is_alive(self):
            return True

    thread = SlowThread()
    device = ScrcpyDevice()
    device._thread = thread

    device.stop()

    assert thread.joined
    assert device._thread is thread


def test_ldplayer_stop_closes_capture_and_persistent_adb_shell():
    camera = _StoppedResource()
    adb = _StoppedResource()
    device = LDPlayerDevice.__new__(LDPlayerDevice)
    device._cam = camera
    device._adb = adb
    device._use_dx = True
    device._present_cache = np.ones((2, 3, 3), dtype=np.uint8)
    device._adb_only = False
    device._stopped = False

    device.stop()

    assert camera.released
    assert adb.stopped
    assert device.last_frame() is None
