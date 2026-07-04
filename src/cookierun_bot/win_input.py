"""Windows-native input for emulators that block `adb shell input` (e.g. BlueStacks).

Capture stays on ADB screencap; only tapping is done here, by mapping a guest-frame
coordinate onto the live emulator window and sending a SendInput mouse click.

`map_guest_to_screen` is pure (no ctypes) so it is unit-testable on any platform;
everything else imports ctypes lazily and is Windows-only at call time.
"""
from __future__ import annotations
import time


def map_guest_to_screen(rect, top_bar, right_bar, guest_w, guest_h, gx, gy):
    """Map a guest-frame pixel (gx,gy) to an absolute screen pixel.

    rect = (left, top, right, bottom) of the emulator window in screen coords.
    top_bar / right_bar = window chrome (title bar height, side toolbar width) in px.
    The guest image is fit (aspect-preserving, letterboxed, centered) into the game
    area = window minus chrome.
    """
    l, t, r, b = rect
    game_w = (r - l) - right_bar
    game_h = (b - t) - top_bar
    guest_ar = guest_w / guest_h
    game_ar = game_w / game_h
    if guest_ar >= game_ar:            # guest wider -> fit by width, letterbox top/bottom
        scale = game_w / guest_w
        off_x = 0.0
        off_y = (game_h - guest_h * scale) / 2.0
    else:                              # guest taller -> fit by height, letterbox sides
        scale = game_h / guest_h
        off_y = 0.0
        off_x = (game_w - guest_w * scale) / 2.0
    sx = l + off_x + gx * scale
    sy = (t + top_bar) + off_y + gy * scale
    return int(round(sx)), int(round(sy))


def set_dpi_aware():
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def find_window(title_substr: str):
    import ctypes
    import ctypes.wintypes
    user32 = ctypes.windll.user32
    found = []

    def _enum(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_substr.lower() in buf.value.lower():
                    found.append(hwnd)
        return True

    proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)(_enum)
    user32.EnumWindows(proc, 0)
    return found[0] if found else None


def get_window_rect(hwnd):
    import ctypes
    import ctypes.wintypes
    r = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r))
    return (r.left, r.top, r.right, r.bottom)


def monitor_of_window(hwnd):
    """Virtual-desktop rect (left, top, right, bottom) of the monitor hosting hwnd."""
    import ctypes
    import ctypes.wintypes

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.wintypes.DWORD),
                    ("rcMonitor", ctypes.wintypes.RECT),
                    ("rcWork", ctypes.wintypes.RECT),
                    ("dwFlags", ctypes.wintypes.DWORD)]

    mon = ctypes.windll.user32.MonitorFromWindow(hwnd, 2)   # MONITOR_DEFAULTTONEAREST
    mi = MONITORINFO()
    mi.cbSize = ctypes.sizeof(MONITORINFO)
    ctypes.windll.user32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcMonitor
    return (r.left, r.top, r.right, r.bottom)


def grab_bbox(bbox):
    """Grab a screen rectangle (left, top, right, bottom) as a BGR ndarray."""
    import numpy as np
    from PIL import ImageGrab
    img = ImageGrab.grab(bbox=bbox, all_screens=True)
    return np.asarray(img)[:, :, ::-1].copy()   # RGB -> BGR


def match_gamearea(guest, win):
    """Locate the guest frame inside a window grab (multi-scale template match).
    Returns ((offset_x, offset_y), (width, height), confidence) in window pixels —
    i.e. where/how big the emulator's game render sits within its host window."""
    import numpy as np
    import cv2
    gh, gw = guest.shape[:2]
    gg = cv2.cvtColor(guest, cv2.COLOR_BGR2GRAY)
    gwn = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)
    best = None
    for s in [x / 1000.0 for x in range(250, 1055, 10)]:
        tw, th = int(gw * s), int(gh * s)
        if tw < 60 or tw >= gwn.shape[1] or th >= gwn.shape[0]:
            continue
        tmpl = cv2.resize(gg, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gwn, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (mx, (ml[0], ml[1]), (tw, th))
    if best is None:
        raise RuntimeError("could not locate emulator game area inside window")
    conf, off, size = best
    return off, size, conf


def foreground(hwnd):
    import ctypes
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    fg = user32.GetForegroundWindow()
    tid_fg = user32.GetWindowThreadProcessId(fg, None)
    tid_me = kernel32.GetCurrentThreadId()
    user32.AttachThreadInput(tid_me, tid_fg, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(tid_me, tid_fg, False)
    time.sleep(0.4)


_LEFT_DOWN = 0x0002
_LEFT_UP = 0x0004


def click(x, y):
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.02)
    user32.mouse_event(_LEFT_DOWN, 0, 0, 0, 0)
    time.sleep(0.03)
    user32.mouse_event(_LEFT_UP, 0, 0, 0, 0)


def hold(x, y, duration_ms):
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.02)
    user32.mouse_event(_LEFT_DOWN, 0, 0, 0, 0)
    time.sleep(max(duration_ms, 30) / 1000.0)
    user32.mouse_event(_LEFT_UP, 0, 0, 0, 0)
