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


class NetworkDevice:
    """Talks to the on-device CR Bridge app over TCP/Wi-Fi: capture via MediaProjection,
    input via AccessibilityService. No ADB, no developer options. Coordinates are in
    captured-frame (phone screen) pixels."""
    def __init__(self, host: str, port: int = 8080, timeout: float = 5.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._sock = None
        self._guest_size: tuple[int, int] | None = None
        # capture size (W,H) and display rotation, from INFO; drives the tap transform.
        self._cap = (0, 0)
        self._rot = 0

    def start(self) -> None:
        import socket
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        self._calibrate_transform()

    def _calibrate_transform(self) -> None:
        """AccessibilityService gestures use the display's NATURAL-orientation coordinate
        space, but MediaProjection captures in the current rotation. Read capture size +
        rotation from INFO and map captured (x,y) -> natural-orientation gesture coords."""
        import re
        info = self.info()
        cap = re.search(r"capture=(\d+)x(\d+)", info)
        rot = re.search(r"rot=(\d+)", info)
        if cap:
            self._cap = (int(cap.group(1)), int(cap.group(2)))  # (width, height)
        if rot:
            self._rot = int(rot.group(1))

    def _to_gesture(self, x, y) -> tuple[int, int]:
        cw, ch = self._cap
        r = self._rot
        if r == 1:      # ROTATION_90 (landscape): gx = ch - cy, gy = cx
            return int(round(ch - y)), int(round(x))
        if r == 3:      # ROTATION_270 (landscape, other way): gx = cy, gy = cw - cx
            return int(round(y)), int(round(cw - x))
        return int(round(x)), int(round(y))   # ROTATION_0 (portrait): identity

    def stop(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("bridge socket closed")
            buf += chunk
        return buf

    def _read_line(self) -> bytes:
        line = b""
        while not line.endswith(b"\n"):
            c = self._sock.recv(1)
            if not c:
                break
            line += c
        return line

    def last_frame(self):
        import numpy as np
        import cv2
        self._sock.sendall(b"FRAME\n")
        n = int.from_bytes(self._recv_exact(4), "big")
        if n == 0:
            return None
        data = self._recv_exact(n)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # BGR
        if img is not None:
            self._guest_size = (img.shape[1], img.shape[0])
        return img

    @property
    def resolution(self) -> tuple[int, int]:
        return self._guest_size or (0, 0)

    def tap(self, x: int, y: int):
        gx, gy = self._to_gesture(x, y)
        self._sock.sendall(f"TAP {gx} {gy}\n".encode())
        return self._read_line().decode(errors="replace").strip()

    def hold(self, x: int, y: int, duration_ms: int):
        gx, gy = self._to_gesture(x, y)
        self._sock.sendall(f"HOLD {gx} {gy} {int(duration_ms)}\n".encode())
        return self._read_line().decode(errors="replace").strip()

    def info(self) -> str:
        """Diagnostics: 'acc=<bool> capture=WxH real=WxH rot=N' from the bridge app."""
        self._sock.sendall(b"INFO\n")
        return self._read_line().decode(errors="replace").strip()

    def global_action(self, name: str):
        """Coordinate-free system action: BACK | HOME | SHADE."""
        self._sock.sendall(f"GLOBAL {name}\n".encode())
        return self._read_line().decode(errors="replace").strip()

    def probe(self, gx: int, gy: int):
        """Draw a calibration dot at gesture coord (gx,gy) (raw, no transform)."""
        self._sock.sendall(f"PROBE {int(gx)} {int(gy)}\n".encode())
        return self._read_line().decode(errors="replace").strip()


class LDPlayerDevice:
    """LDPlayer: fast Windows window-grab capture (resized to guest coords) + adb input.
    LDPlayer allows `adb shell input` (unlike BlueStacks), and window-grab (~13fps) is
    far faster than adb screencap (~1fps at 2560x1440). Self-calibrates the game-area
    rect inside the window once via template match, then re-queries the live window
    position each frame so a moved window still works."""
    def __init__(self, serial: str | None, window_title: str = "LDPlayer"):
        self._adb = AdbDevice(serial)          # adb input + one-time calibration screencap
        self._window_title = window_title
        self._hwnd = None
        self._off = (0, 0)                     # game-area offset within window (px)
        self._ga = (0, 0)                      # game-area size within window (px)
        self._guest = (2560, 1440)             # guest resolution (frame we present)

    def start(self) -> None:
        from . import win_input
        win_input.set_dpi_aware()
        self._hwnd = win_input.find_window(self._window_title)
        if self._hwnd is None:
            raise RuntimeError(f"emulator window not found: '{self._window_title}'")
        guest = self._adb.last_frame()         # pure guest frame via adb screencap
        if guest is None:
            raise RuntimeError("adb screencap returned no frame for calibration")
        self._guest = (guest.shape[1], guest.shape[0])
        win = win_input.grab_bbox(win_input.get_window_rect(self._hwnd))
        off, size, conf = win_input.match_gamearea(guest, win)
        self._off, self._ga = off, size
        self._calib_conf = conf

    def stop(self) -> None:
        pass

    def last_frame(self):
        from . import win_input
        import cv2
        rect = win_input.get_window_rect(self._hwnd)
        gx, gy = rect[0] + self._off[0], rect[1] + self._off[1]
        sub = win_input.grab_bbox((gx, gy, gx + self._ga[0], gy + self._ga[1]))
        if sub is None or sub.size == 0:
            return None
        return cv2.resize(sub, self._guest, interpolation=cv2.INTER_AREA)

    @property
    def resolution(self) -> tuple[int, int]:
        return self._guest

    def tap(self, x: int, y: int) -> None:
        self._adb.tap(x, y)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        self._adb.hold(x, y, duration_ms)


def open_device(cfg) -> Device:
    if cfg.capture_backend == "ldplayer":
        return LDPlayerDevice(cfg.device_serial, cfg.window_title)
    if cfg.capture_backend == "network":
        return NetworkDevice(cfg.phone_host, cfg.phone_port)
    if cfg.capture_backend == "bluestacks":
        return BlueStacksDevice(cfg.device_serial, cfg.window_title,
                                cfg.window_top_bar, cfg.window_right_bar)
    if cfg.capture_backend == "adb":
        return AdbDevice(cfg.device_serial)
    return ScrcpyDevice(cfg.device_serial, cfg.max_fps)
