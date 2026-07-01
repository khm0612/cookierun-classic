from __future__ import annotations
import time
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Device(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def last_frame(self) -> "np.ndarray | None": ...
    @property
    def resolution(self) -> tuple[int, int]: ...
    def tap(self, x: int, y: int) -> None: ...
    def hold(self, x: int, y: int, duration_ms: int) -> None: ...


class ScrcpyDevice:
    """Low-latency capture + control via scrcpy-client."""
    def __init__(self, serial: str | None = None, max_fps: int = 60):
        import scrcpy  # imported lazily so tests without a phone still import the module
        self._scrcpy = scrcpy
        self._client = scrcpy.Client(
            device=serial, max_fps=max_fps, block_frame=True
        )
        self._client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
        self._latest = None

    def _on_frame(self, frame):
        if frame is not None:
            self._latest = frame  # BGR ndarray

    def start(self) -> None:
        self._client.start(threaded=True)

    def stop(self) -> None:
        self._client.stop()

    def last_frame(self):
        return self._latest

    @property
    def resolution(self) -> tuple[int, int]:
        return self._client.resolution

    def tap(self, x: int, y: int) -> None:
        self._client.control.touch(x, y, self._scrcpy.ACTION_DOWN)
        self._client.control.touch(x, y, self._scrcpy.ACTION_UP)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        self._client.control.touch(x, y, self._scrcpy.ACTION_DOWN)
        time.sleep(duration_ms / 1000.0)
        self._client.control.touch(x, y, self._scrcpy.ACTION_UP)


class AdbDevice:
    """Slower fallback via adbutils. Fine for menus; too slow for in-run reactions."""
    def __init__(self, serial: str | None = None):
        import adbutils
        self._dev = (adbutils.adb.device(serial=serial) if serial
                     else adbutils.adb.device_list()[0])

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def last_frame(self):
        img = self._dev.screenshot()          # PIL.Image (RGB)
        return np.asarray(img)[:, :, ::-1].copy()  # -> BGR ndarray

    @property
    def resolution(self) -> tuple[int, int]:
        w, h = self._dev.window_size()
        return (w, h)

    def tap(self, x: int, y: int) -> None:
        self._dev.click(x, y)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        self._dev.swipe(x, y, x, y, duration_ms / 1000.0)


class BlueStacksDevice:
    """Capture via ADB screencap; input via Windows SendInput mapped onto the live
    emulator window. Used for BlueStacks, whose adbd serves screencap but refuses
    `adb shell input`. The cursor is moved to each tap, so the emulator window must
    stay visible/foreground while farming.
    """
    def __init__(self, serial: str | None, window_title: str,
                 top_bar: int = 40, right_bar: int = 40):
        self._adb = AdbDevice(serial)          # capture only (never .tap/.hold)
        self._window_title = window_title
        self._top_bar = top_bar
        self._right_bar = right_bar
        self._guest_size: tuple[int, int] | None = None
        self._hwnd = None

    def start(self) -> None:
        from . import win_input
        win_input.set_dpi_aware()
        self._hwnd = win_input.find_window(self._window_title)
        if self._hwnd is None:
            raise RuntimeError(f"emulator window not found: '{self._window_title}'")
        win_input.foreground(self._hwnd)

    def stop(self) -> None:
        pass

    def last_frame(self):
        frame = self._adb.last_frame()
        if frame is not None:
            self._guest_size = (frame.shape[1], frame.shape[0])
        return frame

    @property
    def resolution(self) -> tuple[int, int]:
        return self._guest_size or (1920, 1080)

    def _to_screen(self, gx: int, gy: int) -> tuple[int, int]:
        from . import win_input
        if self._hwnd is None:
            raise RuntimeError("device not started; call start() first")
        rect = win_input.get_window_rect(self._hwnd)
        gw, gh = self._guest_size or (1920, 1080)
        return win_input.map_guest_to_screen(
            rect, self._top_bar, self._right_bar, gw, gh, gx, gy)

    def tap(self, x: int, y: int) -> None:
        from . import win_input
        sx, sy = self._to_screen(x, y)
        win_input.click(sx, sy)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        from . import win_input
        sx, sy = self._to_screen(x, y)
        win_input.hold(sx, sy, duration_ms)


def open_device(cfg) -> Device:
    if cfg.capture_backend == "bluestacks":
        return BlueStacksDevice(cfg.device_serial, cfg.window_title,
                                cfg.window_top_bar, cfg.window_right_bar)
    if cfg.capture_backend == "adb":
        return AdbDevice(cfg.device_serial)
    return ScrcpyDevice(cfg.device_serial, cfg.max_fps)
