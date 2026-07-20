from __future__ import annotations
import time
from typing import Protocol, runtime_checkable
import numpy as np


def select_adb_serial(requested: str, devices: list[str]) -> tuple[str, str]:
    requested = requested.strip()
    if requested and requested in devices:
        return requested, "ready"
    if requested:
        return requested, "device missing"
    if devices:
        return devices[0], "ready"
    return "", "no devices"


def ready_adb_serials() -> list[str]:
    import adbutils
    return [dev.serial for dev in adbutils.adb.device_list()]


def resolve_adb_serial(requested: str | None) -> str | None:
    selected, status = select_adb_serial(requested or "", ready_adb_serials())
    if status == "ready" and selected:
        return selected
    return requested or None


def scrcpy_server_jar_path(jar_name: str = "scrcpy-server-v1.24.jar") -> str:
    import importlib.util
    import os
    spec = importlib.util.find_spec("scrcpy")
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locations:
        raise RuntimeError("scrcpy-client is installed without the scrcpy package")
    jar = os.path.join(next(iter(locations)), jar_name)
    if not os.path.exists(jar):
        raise RuntimeError(f"scrcpy server jar not found: {jar}")
    return jar


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
    """Fast capture via scrcpy's display encoder (H.264 over adb) + adb input.

    Kept as a support backend for devices where the H.264 display stream is the
    cleanest source. For the current LDPlayer farm path, LDPlayerDevice is the
    preferred high-FPS backend.

    Self-contained (drives scrcpy-server directly over adb, no scrcpy.Client) so it
    works with adbutils 2.x, and the decode loop tolerates a partial/corrupt NAL
    (av raises InvalidDataError) instead of letting one bad packet kill the stream.
    Capture is full-resolution, so frame pixels share the device's tap coordinate
    space and no rescaling is needed. Input goes through adb (proven on LDPlayer).
    """
    _JAR_NAME = "scrcpy-server-v1.24.jar"

    def __init__(self, serial: str | None = None, max_fps: int = 0,
                 max_size: int = 0, bitrate: int = 8_000_000):
        self._serial = serial
        self._max_fps = max_fps
        self._max_size = max_size
        self._bitrate = bitrate
        self._adb = None                       # input fallback (adb `input`)
        self._dev = None                       # adbutils device (capture channel)
        self._vs = None                        # video socket
        self._cs = None                        # control socket (instant touch injection)
        self._server = None                    # server shell stream
        self._thread = None
        self._alive = False                    # intent: capture should be running
        self._dead = False                     # the decode thread has exited
        self._last_restart = 0.0               # monotonic time of the last auto-restart
        self._latest = None
        self._res = (0, 0)
        self._frame_count = 0                  # diag: total frames decoded
        self._hold_until = 0.0                 # one virtual finger: drop overlapping gestures
        import threading
        self._lock = threading.Lock()
        self._cs_lock = threading.Lock()       # serialize control-socket writes
        self._frame_event = threading.Event()  # set on every decoded frame (streaming loop)

    def _recv_exact(self, n: int) -> bytes:
        """recv() may return fewer bytes than asked; the fixed-width scrcpy header must be
        read in full or struct.unpack crashes / the H.264 stream desyncs."""
        buf = b""
        while len(buf) < n:
            chunk = self._vs.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("scrcpy: video socket closed during header")
            buf += chunk
        return buf

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("scrcpy decoder thread is still running")
        import struct
        import threading
        import adbutils
        if not hasattr(adbutils, "_AdbStreamConnection"):
            adbutils._AdbStreamConnection = object   # scrcpy jar path helper import shim
        from adbutils import Network, AdbError

        self._serial = resolve_adb_serial(self._serial)
        self._adb = AdbDevice(self._serial)
        self._dev = (adbutils.adb.device(serial=self._serial) if self._serial
                     else adbutils.adb.device_list()[0])
        jar = scrcpy_server_jar_path(self._JAR_NAME)
        self._dev.sync.push(jar, "/data/local/tmp/" + self._JAR_NAME)
        cmds = [
            f"CLASSPATH=/data/local/tmp/{self._JAR_NAME}",
            "app_process", "/", "com.genymobile.scrcpy.Server", "1.24",
            "log_level=info", f"bit_rate={self._bitrate}", f"max_size={self._max_size}",
            f"max_fps={self._max_fps}", "lock_video_orientation=-1",
            "tunnel_forward=true", "control=false", "display_id=0",
            "show_touches=false", "stay_awake=false", "clipboard_autosync=false",
        ]
        self._server = self._dev.shell(cmds, stream=True)
        try:                                   # anything past here leaks the server on error
            self._server.read(10)              # wait for the server to come up
            for _ in range(30):
                try:
                    self._vs = self._dev.create_connection(Network.LOCAL_ABSTRACT, "scrcpy")
                    break
                except AdbError:
                    time.sleep(0.1)
            if self._vs is None:
                raise RuntimeError("scrcpy: could not connect video socket")
            # NOTE: control socket disabled. The scrcpy v1.24 control-socket touch-event
            # wire format we tried registered NOTHING on LDPlayer (socket send succeeds but
            # the server drops the event), and because send() "succeeds" there was no adb
            # fallback -> zero taps landed. Input goes through adb `input touchscreen swipe`
            # (proven reliable on LDPlayer). Streaming CAPTURE via wait_frame() is unaffected.
            self._cs = None
            if self._recv_exact(1) != b"\x00":
                raise RuntimeError("scrcpy: missing dummy byte")
            self._recv_exact(64)               # device name
            w, h = struct.unpack(">HH", self._recv_exact(4))
            self._res = (w, h)
            self._dead = False
            self._alive = True
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
            for _ in range(50):                # let the first frame decode (~keyframe)
                if self._latest is not None:
                    break
                time.sleep(0.1)
        except BaseException:
            self.stop()                        # close the socket + kill the orphan server
            raise

    def _stream_loop(self) -> None:
        from av.codec import CodecContext
        codec = CodecContext.create("h264", "r")
        try:
            while self._alive:
                try:
                    data = self._vs.recv(0x10000)
                except OSError:
                    break
                if not data:
                    break                      # socket EOF: server died / device gone
                try:
                    packets = codec.parse(data)
                except Exception:
                    continue
                for packet in packets:
                    try:
                        for frame in codec.decode(packet):
                            arr = frame.to_ndarray(format="bgr24")   # BGR, full device res
                            with self._lock:
                                if not self._alive:
                                    break
                                self._latest = arr
                                self._res = (arr.shape[1], arr.shape[0])
                                self._frame_count += 1
                            self._frame_event.set()   # wake the streaming decision loop
                    except Exception:
                        continue               # skip a bad packet/decode, keep streaming
        finally:
            self._dead = True                  # let last_frame() detect the thread exited
            self._frame_event.set()            # unblock any wait_frame() caller

    def wait_frame(self, timeout: float = 0.5):
        """Streaming read: block until a NEW frame is decoded (or timeout), then return
        the latest frame. scrcpy only sends frames on screen change, so a timeout on a
        static screen returns the same frame — callers see identical-frame semantics."""
        self._frame_event.wait(timeout)
        self._frame_event.clear()
        return self.last_frame()

    def stop(self) -> None:
        self._alive = False
        if self._adb is not None:
            try:
                self._adb.stop()               # close the persistent input shell
            except Exception:
                pass
        for sock in (self._vs, self._cs, self._server):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None:
            import threading
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        with self._lock:
            self._latest = None
            self._res = (0, 0)
        self._frame_event.clear()
        self._vs = None
        self._cs = None
        self._server = None
        self._thread = thread if thread is not None and thread.is_alive() else None

    def _restart(self) -> None:
        """The decode thread exited unexpectedly (socket EOF / emulator hiccup) while we
        still want capture. Re-establish it, rate-limited so a hard failure can't spin."""
        now = time.monotonic()
        if now - self._last_restart < 3.0:
            return
        self._last_restart = now
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.start()
        except Exception:
            # stop() cleared _alive; re-arm the heal intent so the NEXT last_frame()
            # retries after the rate-limit window instead of serving a stale frame forever.
            self._alive = True
            self._dead = True

    def last_frame(self):
        if self._alive and self._dead:         # thread died but we want it -> self-heal
            self._restart()
        with self._lock:
            return self._latest

    @property
    def resolution(self) -> tuple[int, int]:
        return self._res

    # -- input ------------------------------------------------------------------
    # scrcpy control-socket touch injection (v1.24 TYPE_INJECT_TOUCH_EVENT):
    # type u8=2, action u8 (0 down / 1 up), pointerId u64, x i32, y i32,
    # screenW u16, screenH u16, pressure u16, buttons u32. screen size MUST match
    # the video size or the server silently drops the event. Instant (no adb-shell
    # round-trip) and holds are non-blocking. Falls back to adb `input` if the
    # control socket is unavailable or a write fails.

    def _send_touch(self, action: int, x: int, y: int) -> bool:
        if self._cs is None:
            return False
        import struct
        w, h = self._res
        msg = struct.pack(">BBQiiHHHI", 2, action, 0, int(x), int(y), w, h,
                          0xFFFF if action == 0 else 0, 0)
        try:
            with self._cs_lock:
                self._cs.send(msg)
            return True
        except Exception:
            self._cs = None                    # broken pipe: fall back to adb input
            return False

    def _finger_busy(self) -> bool:
        # one virtual finger: while a hold is in flight, extra gestures are dropped
        # (prevents 90fps slide-spam from stacking overlapping down/up events)
        return time.monotonic() < self._hold_until

    def tap(self, x: int, y: int) -> None:
        if self._finger_busy():
            return
        if self._send_touch(0, x, y):
            self._hold_until = time.monotonic() + 0.07
            import threading
            threading.Timer(0.06, self._send_touch, args=(1, x, y)).start()
            return
        if self._adb is None:
            self._adb = AdbDevice(self._serial)
        self._adb.tap(x, y)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        if self._finger_busy():
            return
        if self._send_touch(0, x, y):
            self._hold_until = time.monotonic() + duration_ms / 1000.0
            import threading
            threading.Timer(duration_ms / 1000.0, self._send_touch, args=(1, x, y)).start()
            return
        if self._adb is None:
            self._adb = AdbDevice(self._serial)
        self._adb.hold(x, y, duration_ms)

    def back(self) -> None:
        if self._adb is None:
            self._adb = AdbDevice(self._serial)
        self._adb.back()


class AdbDevice:
    """ADB device with a PERSISTENT shell for low-latency input.

    Measured on LDPlayer: adbutils `.shell("input ...")` spends ~116 ms PER TAP setting up
    a fresh adb exec each time — that lag is a direct cause of late dodges (the cookie hits
    obstacles the detector already flagged). A single long-lived `adb shell` fed via stdin
    turns each tap into a ~0 ms fire-and-forget write; the on-device `input` runs async, so
    the decision loop never blocks on I/O. adbutils remains the capture channel + the
    fallback if the persistent shell can't be spawned or its pipe breaks.
    """
    def __init__(self, serial: str | None = None):
        import threading
        import adbutils
        serial = resolve_adb_serial(serial)
        self._dev = (adbutils.adb.device(serial=serial) if serial
                     else adbutils.adb.device_list()[0])
        self._serial = self._dev.serial
        self._shell = None                    # persistent `adb shell` Popen
        self._shell_lock = threading.Lock()

    def start(self) -> None:
        pass

    def _adb_exe(self) -> str:
        try:
            import adbutils
            return adbutils.adb_path()
        except Exception:
            return "adb"

    def _ensure_shell(self):
        if self._shell is not None and self._shell.poll() is None:
            return self._shell
        import subprocess
        try:
            self._shell = subprocess.Popen(
                [self._adb_exe(), "-s", self._serial, "shell"],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except Exception:
            self._shell = None
        return self._shell

    def _send(self, line: str) -> bool:
        """Fire-and-forget a shell command down the persistent pipe. One reopen retry on a
        broken pipe; returns False so the caller can fall back to adbutils."""
        with self._shell_lock:
            for _ in range(2):
                sh = self._ensure_shell()
                if sh is None or sh.stdin is None:
                    return False
                try:
                    sh.stdin.write(line + "\n")
                    sh.stdin.flush()
                    return True
                except (BrokenPipeError, OSError):
                    self._shell = None        # dropped — reopen and retry once
        return False

    def reset_shell(self) -> None:
        """Kill + respawn the persistent input shell. Fire-and-forget writes can't detect a
        shell whose REMOTE adb session died while the local adb.exe stays alive (poll() looks
        healthy, every input line is silently discarded — observed live 2026-07-04: a wedged
        boost screen where verified Play taps never landed for 10+ min). Callers that detect
        repeated NO-EFFECT taps should call this before retrying."""
        with self._shell_lock:
            if self._shell is not None:
                try:
                    self._shell.kill()
                except Exception:
                    pass
                self._shell = None

    def stop(self) -> None:
        with self._shell_lock:
            if self._shell is not None:
                try:
                    self._shell.stdin.write("exit\n")
                    self._shell.stdin.flush()
                    self._shell.wait(timeout=2)
                except Exception:
                    try:
                        self._shell.kill()
                    except Exception:
                        pass
                self._shell = None

    def last_frame(self):
        img = self._dev.screenshot()          # PIL.Image (RGB)
        return np.asarray(img)[:, :, ::-1].copy()  # -> BGR ndarray

    @property
    def resolution(self) -> tuple[int, int]:
        w, h = self._dev.window_size()
        return (w, h)

    def tap(self, x: int, y: int) -> None:
        # A 70ms same-point touchscreen swipe, not an instant tap: on some LDPlayer
        # boots default-source `input tap` stops registering IN-GAME entirely, and even
        # instant touchscreen taps get dropped by stale/restored modals — a press that
        # spans several game frames survives both (verified live 2026-07-03).
        line = f"input touchscreen swipe {int(x)} {int(y)} {int(x)} {int(y)} 70"
        if not self._send(line):
            self._dev.shell(line)             # fallback

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        line = ("input touchscreen swipe "
                f"{int(x)} {int(y)} {int(x)} {int(y)} {int(duration_ms)}")
        if not self._send(line):
            self._dev.shell(line)

    def press(self, x: int, y: int) -> None:
        # true touch-DOWN: the finger stays down until release() — variable-length slides.
        # Instant command (no duration), so it never queues backlog in the shell pipe the
        # way a long `input swipe` does. Verified on the LDPlayer Android 14 image.
        line = f"input motionevent DOWN {int(x)} {int(y)}"
        if not self._send(line):
            self._dev.shell(line)

    def release(self, x: int, y: int) -> None:
        line = f"input motionevent UP {int(x)} {int(y)}"
        if not self._send(line):
            self._dev.shell(line)

    def back(self) -> None:
        # BACK dismisses tap-deaf restored modals (acts as OK/close); keyevents always
        # register even when touch injection is being swallowed.
        if not self._send("input keyevent 4"):
            self._dev.shell("input keyevent 4")


class BridgeCommandError(RuntimeError):
    pass


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
    def __init__(self, host: str, port: int = 8080, timeout: float = 5.0,
                 token: str | None = None):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._token = token
        self._sock = None
        self._guest_size: tuple[int, int] | None = None

    def start(self) -> None:
        import os
        import socket
        # ponytail: the bridge token is session-only, so an environment variable is enough.
        token = (self._token or os.environ.get("COOKIERUN_BRIDGE_TOKEN", "")).strip()
        if not token:
            raise RuntimeError(
                "network bridge requires COOKIERUN_BRIDGE_TOKEN from the phone app")
        if any(ch.isspace() for ch in token):
            raise ValueError("network bridge token must not contain whitespace")
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        try:
            if self._send_line(f"AUTH {token}\n") != "OK":
                raise BridgeCommandError("bridge authentication failed")
        except BaseException:
            self.stop()
            raise

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

    def _send_line(self, line: str) -> str:
        if self._sock is None:
            raise ConnectionError("bridge socket is not connected")
        self._sock.sendall(line.encode())
        reply = self._read_line().decode(errors="replace").strip()
        if not reply:
            raise ConnectionError("bridge closed before replying")
        return reply

    def _require_gesture_ok(self, reply: str) -> None:
        status, _, detail = reply.partition(" ")
        if status != "OK":
            raise BridgeCommandError(reply)
        if detail and detail != "completed":
            raise BridgeCommandError(reply)

    def last_frame(self):
        import numpy as np
        import cv2
        if self._sock is None:
            raise ConnectionError("bridge socket is not connected")
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
        self._require_gesture_ok(self._send_line(f"TAP {int(round(x))} {int(round(y))}\n"))

    def hold(self, x: int, y: int, duration_ms: int):
        self._require_gesture_ok(
            self._send_line(f"HOLD {int(round(x))} {int(round(y))} {int(duration_ms)}\n"))

    def info(self) -> str:
        """Diagnostics: 'acc=<bool> capture=WxH real=WxH rot=N' from the bridge app."""
        return self._send_line("INFO\n")

    def global_action(self, name: str):
        """Coordinate-free system action: BACK | HOME | SHADE."""
        return self._send_line(f"GLOBAL {name}\n")

    def probe(self, gx: int, gy: int):
        """Draw a calibration dot at gesture coord (gx,gy) (raw, no transform)."""
        return self._send_line(f"PROBE {int(gx)} {int(gy)}\n")


class LDPlayerDevice:
    """LDPlayer: GPU window-grab capture (dxcam / DXGI Desktop Duplication, ~100+fps) with a
    GDI fallback, plus adb input. scrcpy's H.264 path collapsed to ~1fps in-run on this box,
    and the CPU GDI grab caps at ~16fps; dxcam reads the composited window straight off the
    GPU at the monitor refresh rate, which is what finally gives the bot enough frames to dodge.

    Every frame is presented in a canonical 2560x1440 space so the templates/regions/gestures
    (all calibrated at 2560) keep working no matter the emulator's real render resolution;
    taps are scaled from that canonical space down to the device's actual input resolution
    (adb `input` coordinates are in real device pixels). Self-calibrates the game-area rect
    inside the window once, then re-queries the live window position each frame so a moved
    window still works. Needs the LDPlayer window visible/foreground while farming."""
    _PRESENT = (2560, 1440)                    # canonical frame space the bot reasons in

    def __init__(self, serial: str | None, window_title: str = "LDPlayer"):
        self._adb = AdbDevice(serial)          # adb input + one-time calibration screencap
        self._window_title = window_title
        self._hwnd = None
        self._off = (0, 0)                     # game-area offset within window (px)
        self._ga = (0, 0)                      # game-area size within window (px)
        self._input_res = (2560, 1440)         # device's real display res (for tap scaling)
        self._adb_only = False
        self._cam = None                       # dxcam camera (GPU capture)
        self._use_dx = False                   # dxcam verified to see the window (not black)
        self._present_cache = None             # last presented frame (dxcam returns None if unchanged)
        self._stopped = True

    def start(self) -> None:
        from . import win_input
        win_input.set_dpi_aware()
        self._hwnd = win_input.find_window(self._window_title)
        if self._hwnd is None:
            raise RuntimeError(f"emulator window not found: '{self._window_title}'")
        win_input.foreground(self._hwnd)       # restore/raise: window-grab needs it visible+topmost
        guest = self._adb.last_frame()         # pure guest frame via adb screencap
        if guest is None:
            raise RuntimeError("adb screencap returned no frame for calibration")
        self._stopped = False
        self._input_res = (guest.shape[1], guest.shape[0])   # e.g. 1600x900 or 2560x1440
        rect = win_input.get_window_rect(self._hwnd)
        if rect[0] <= -30000 or (rect[2] - rect[0]) < 320 or (rect[3] - rect[1]) < 240:
            self._adb_only = True              # minimized/offscreen: screen-grab cannot see it
            return
        win = win_input.grab_bbox(rect)
        try:
            off, size, conf = win_input.match_gamearea(guest, win)
        except RuntimeError:
            self._adb_only = True              # slower, but still read-only and reliable
            return
        self._off, self._ga = off, size
        self._calib_conf = conf
        self._init_dxcam()

    def _init_dxcam(self) -> None:
        """Create the GPU camera on the OUTPUT THAT HOSTS THE WINDOW and verify its frames
        against a GDI grab of the same region. dxcam regions are in output-LOCAL coords, and
        on a multi-monitor desktop the window may live on any output — capturing output 0
        unconditionally silently served wrong/stale content (live-debugged 2026-07-04: the
        LDPlayer window sat on the second monitor at x=3668 while dxcam watched the primary,
        so capture quietly fell back to ~12fps GDI and starved the policy)."""
        self._use_dx = False
        try:
            import dxcam
            import cv2
            import numpy as np
            from . import win_input
            mon = win_input.monitor_of_window(self._hwnd)     # (l, t, r, b) virtual coords
            self._mon_origin = (mon[0], mon[1])
            gdi = win_input.grab_bbox(self._virtual_region()) # ground truth
            gg = cv2.resize(cv2.cvtColor(gdi, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
            gg -= gg.mean()
            n_out = len([ln for ln in dxcam.output_info().splitlines() if "Output" in ln])
            best = (0.0, None)                                 # (corr, cam)
            for idx in range(max(n_out, 1)):
                cam = None
                try:
                    cam = dxcam.create(output_idx=idx, output_color="BGR")
                    if cam is None:
                        continue
                    self._cam = cam
                    # a fresh duplicator yields no frame until its output CHANGES; a static
                    # screen would fail validation forever — jiggle the cursor over that
                    # monitor (no clicks) to force a composition, and retry briefly.
                    fr = None
                    for attempt in range(20):
                        fr = self._grab_dx()
                        if fr is not None and fr.size:
                            break
                        if attempt == 6:
                            import ctypes
                            import ctypes.wintypes
                            pt = ctypes.wintypes.POINT()
                            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                            ctypes.windll.user32.SetCursorPos(mon[0] + 8, mon[1] + 8)
                            time.sleep(0.05)
                            ctypes.windll.user32.SetCursorPos(pt.x, pt.y)
                        time.sleep(0.1)
                    if fr is None or fr.size == 0 or fr.shape[:2] != gdi.shape[:2]:
                        raise ValueError("no frame / size mismatch")
                    # structural match vs GDI truth: normalized correlation on mean-subtracted
                    # grayscale — GDI/DXGI color pipelines differ (gamma/night-light), so an
                    # absolute pixel diff false-rejects the CORRECT output (measured: right
                    # output diff 31, wrong output 83).
                    fg = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
                    fg -= fg.mean()
                    denom = float(np.sqrt((fg * fg).sum() * (gg * gg).sum())) or 1.0
                    corr = float((fg * gg).sum()) / denom
                    if corr > best[0]:
                        if best[1] is not None:
                            try: best[1].release()
                            except Exception: pass
                        best = (corr, cam)
                        continue                               # keep this cam as candidate
                    raise ValueError(f"lower corr {corr:.2f}")
                except Exception:
                    if cam is not None and cam is not best[1]:
                        try: cam.release()
                        except Exception: pass
                    self._cam = None
            if best[1] is not None and best[0] > 0.6:
                self._cam = best[1]
                self._use_dx = True
                # seed the freshness clock so the FIRST static-screen wait_frame() timeout
                # can't trip the dead-duplicator heal (getattr default 0.0 => always stale).
                self._last_fresh = time.monotonic()
                return
            if best[1] is not None:
                try: best[1].release()
                except Exception: pass
        except Exception:
            self._cam = None
            self._use_dx = False

    def _virtual_region(self) -> tuple[int, int, int, int]:
        from . import win_input
        rect = win_input.get_window_rect(self._hwnd)
        gx, gy = rect[0] + self._off[0], rect[1] + self._off[1]
        return (gx, gy, gx + self._ga[0], gy + self._ga[1])

    def _region(self) -> tuple[int, int, int, int]:
        """Window game-area in the dxcam output's LOCAL coordinate space."""
        gx, gy, gx2, gy2 = self._virtual_region()
        ox, oy = getattr(self, "_mon_origin", (0, 0))
        return (gx - ox, gy - oy, gx2 - ox, gy2 - oy)

    def _grab_dx(self):
        if self._cam is None:
            return None
        try:
            return self._cam.grab(region=self._region())   # BGR ndarray, or None if unchanged
        except Exception:
            return None

    def stop(self) -> None:
        try:
            if self._cam is not None:
                self._cam.release()
        except Exception:
            pass
        self._cam = None
        self._use_dx = False
        self._present_cache = None
        self._stopped = True
        try:
            self._adb.stop()
        except Exception:
            pass

    def last_frame(self):
        if self._stopped:
            return None
        import cv2
        if self._adb_only:
            raw = self._adb.last_frame()         # canonicalize to _PRESENT like every other path:
            if raw is None or raw.size == 0:     # the resolution property reports 2560x1440, and
                return None                      # abs-pixel consumers (hp_frac, HUD templates) rely
            return cv2.resize(raw, self._PRESENT, interpolation=cv2.INTER_LINEAR)  # on that space
        raw = self._grab_dx() if self._use_dx else None
        if raw is None and self._use_dx and self._present_cache is not None:
            return self._present_cache         # dxcam: None == unchanged -> serve cached present
        if raw is None:                        # GDI fallback (or first frame before dxcam warms)
            from . import win_input
            raw = win_input.grab_bbox(self._virtual_region())
        if raw is None or raw.size == 0:
            return None
        out = cv2.resize(raw, self._PRESENT, interpolation=cv2.INTER_LINEAR)
        self._present_cache = out
        return out

    def wait_frame(self, timeout: float = 0.5):
        """Streaming read: block until the screen CHANGES (dxcam returns a frame only on
        change), then return it; on timeout return the latest frame. Same semantics as
        ScrcpyDevice.wait_frame, so the recorder / play loop pace on real new frames
        instead of spinning on the cached one. GDI/adb fallbacks have no change signal —
        there a plain grab IS the pace (~16fps GDI), so just return it.

        SELF-HEAL: a DXGI duplicator silently dies on desktop events (observed live after
        a minutes-long idle wait: grab() returned None forever while the game visibly ran,
        so the bot played BLIND off the cached frame for whole runs). If no fresh frame
        arrives for >3s, re-create the duplicator; if that fails, drop to GDI."""
        if self._adb_only or not self._use_dx:
            return self.last_frame()
        import cv2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self._grab_dx()
            if raw is not None and raw.size:
                out = cv2.resize(raw, self._PRESENT, interpolation=cv2.INTER_LINEAR)
                self._present_cache = out
                self._last_fresh = time.monotonic()
                return out
            time.sleep(0.002)
        if time.monotonic() - getattr(self, "_last_fresh", 0.0) > 3.0:
            try:
                if self._cam is not None:
                    self._cam.release()
            except Exception:
                pass
            self._cam = None
            self._use_dx = False
            self._init_dxcam()                 # revalidates against GDI ground truth
            self._last_fresh = time.monotonic()   # rate-limit re-init attempts
        return self._present_cache if self._use_dx else self.last_frame()

    def nav_frame(self):
        """Sharp guest frame via adb screencap (~0.5s) for MENU/template navigation: the
        window-grab path softens detail (game rendered at ~62% window scale then upscaled),
        dropping template scores ~0.10 below their calibrated thresholds. Slow but menus are
        static; in-run reads keep using last_frame/wait_frame."""
        import cv2
        f = self._adb.last_frame()
        if f is not None and (f.shape[1], f.shape[0]) != self._PRESENT:
            f = cv2.resize(f, self._PRESENT, interpolation=cv2.INTER_LINEAR)
        return f

    @property
    def resolution(self) -> tuple[int, int]:
        return self._PRESENT

    def _scale_tap(self, x: int, y: int) -> tuple[int, int]:
        iw, ih = self._input_res
        pw, ph = self._PRESENT
        return int(round(x * iw / pw)), int(round(y * ih / ph))

    def tap(self, x: int, y: int) -> None:
        ax, ay = self._scale_tap(x, y)
        self._adb.tap(ax, ay)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        ax, ay = self._scale_tap(x, y)
        self._adb.hold(ax, ay, duration_ms)

    def press(self, x: int, y: int) -> None:
        ax, ay = self._scale_tap(x, y)
        self._adb.press(ax, ay)

    def release(self, x: int, y: int) -> None:
        ax, ay = self._scale_tap(x, y)
        self._adb.release(ax, ay)

    def back(self) -> None:
        self._adb.back()

    def reset_shell(self) -> None:
        self._adb.reset_shell()


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
