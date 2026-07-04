from cookierun_bot.device import resolve_adb_serial, scrcpy_server_jar_path


def test_scrcpy_server_jar_path_uses_package_location(monkeypatch, tmp_path):
    pkg = tmp_path / "scrcpy"
    pkg.mkdir()
    jar = pkg / "server.jar"
    jar.write_bytes(b"jar")

    class Spec:
        submodule_search_locations = [str(pkg)]

    monkeypatch.setattr("importlib.util.find_spec", lambda name: Spec if name == "scrcpy" else None)

    assert scrcpy_server_jar_path("server.jar") == str(jar)


def test_resolve_adb_serial_switches_stale_single_device(monkeypatch):
    monkeypatch.setattr("cookierun_bot.device.ready_adb_serials", lambda: ["127.0.0.1:5555"])

    assert resolve_adb_serial("emulator-5554") == "127.0.0.1:5555"


def test_resolve_adb_serial_keeps_requested_when_ambiguous(monkeypatch):
    monkeypatch.setattr("cookierun_bot.device.ready_adb_serials", lambda: ["a", "b"])

    assert resolve_adb_serial("missing") == "missing"
