"""Shared low-level helpers for the farm loop: frame reads, timing/stop control,
template polling, result-screen reading, and the boost-gate dataclasses.

Leaf module — imported by farm_cards, farm_boosts, and farm; imports nothing from them."""
from __future__ import annotations
from dataclasses import dataclass
import os
import time

import numpy as np

from .detect import read_results, read_int

# ABSOLUTE path to monitor.py's `card_active` flag (this file is src/cookierun_bot/farm_common.py,
# so three parents up == repo root). Anchored to __file__ NOT cwd so the card-BACK veto works no
# matter where the farm is launched from — a cwd-relative read silently no-ops under any non-root
# launch (adversarial review, 2026-07-05).
_CARD_FLAG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "data", "_selffarm", "card_active")

__all__ = [
    "_nav_read", "_boost_read_fast", "_diff", "_snapshot", "_stop_requested", "_sleep_interruptible",
    "_sleep_remaining", "_wait_for_change", "wait_for_result_frame", "read_run_result",
    "read_wallet",
    "_scrolling", "_in_run", "_tap_template", "_visible_safe_action", "_safe_to_back",
    "_tile_checked", "_find_stable", "_tile_checked_stable",
    "BoostResult", "BoostTileStatus", "BoostGateStatus",
    "_BUTTONS", "_RESULT_BUTTON", "_SAFE_ACTIONS",
]


_BUTTONS = ("play", "ok", "openall", "confirm", "confirm2")
_RESULT_BUTTON = "ok"


_SAFE_ACTIONS = (
    ("cardgame", 0.80, "present"),
    ("boostprompt", 0.75, "present"),
    ("multibtn", 0.80, "present"),
    ("openall", 0.82, "find"),
    ("confirm", 0.82, "find"),
    ("confirm2", 0.82, "find"),
    ("ok", 0.82, "find"),
    ("close", 0.82, "find"),
    ("close2", 0.82, "find"),
    ("play", 0.80, "find"),
)


@dataclass(frozen=True)
class BoostResult:
    active: bool
    spent: int = 0


@dataclass(frozen=True)
class BoostTileStatus:
    name: str
    visible: bool
    checked: bool


@dataclass(frozen=True)
class BoostGateStatus:
    required_tiles: tuple[BoostTileStatus, ...]
    double_coin_banner: bool
    random_boost_button: bool
    pick_boosts_dialog: bool
    multi_buy_button: bool

    @property
    def required_tiles_checked(self) -> bool:
        return all(tile.visible and tile.checked for tile in self.required_tiles)

    @property
    def ready_to_play(self) -> bool:
        return self.required_tiles_checked and self.double_coin_banner


def _nav_read(dev):
    """Frame for MENU/boost-gate template decisions. Window-grab capture softens detail
    enough to drop template scores ~0.10 below their calibrated thresholds (play: 0.898
    sharp vs 0.777 window-grab), so devices exposing nav_frame() (sharp adb screencap)
    get it for navigation; in-run gameplay stays on the fast last_frame/wait_frame path."""
    nav = getattr(dev, "nav_frame", None)
    if nav is not None:
        try:
            f = nav()
            if f is not None:
                return f
        except Exception:
            pass
    return dev.last_frame()


def _boost_read_fast(dev):
    """A FAST frame (dxcam last_frame, ~3ms) for boost-gate BADGE + button verification. The
    sharp adb nav grab is ~880ms; the green-check badge (tilecheck) and the Multi buttons score
    within ~0.02 of it on the fast frame (validated live on the boost screen), so the common
    'tiles already checked' path needs no adb screencap. Falls back to the sharp read if no fast
    frame is available. Decisions that need a below-0.80 ICON match still use the sharp _nav_read."""
    lf = getattr(dev, "last_frame", None)
    if lf is not None:
        try:
            f = lf()
            if f is not None:
                return f
        except Exception:
            pass
    return _nav_read(dev)


def _diff(a, b) -> float:
    if a.shape != b.shape:
        return 255.0                 # size/orientation changed => treat as a big change
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def _snapshot(frame, max_w: int = 160, max_h: int = 90):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    sy = max(1, h // max_h)
    sx = max(1, w // max_w)
    return frame[::sy, ::sx].copy()


def _stop_requested(should_stop=None) -> bool:
    return bool(should_stop and should_stop())


def _sleep_interruptible(seconds: float, should_stop=None,
                         sleep=time.sleep, step_s: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not _stop_requested(should_stop):
        sleep(min(step_s, end - time.monotonic()))


def _sleep_remaining(start: float, period_s: float, sleep=time.sleep) -> None:
    remaining = period_s - (time.monotonic() - start)
    if remaining > 0:
        sleep(remaining)


def _wait_for_change(dev, before, timeout_s: float = 1.0,
                     poll_s: float = 0.1, sleep=time.sleep,
                     now=time.monotonic, should_stop=None) -> None:
    """Wait only until the screen starts changing; no fixed post-tap delay."""
    before_snap = _snapshot(before)
    deadline = now() + timeout_s
    while now() < deadline and not _stop_requested(should_stop):
        after = _nav_read(dev)
        after_snap = _snapshot(after)
        if after_snap is not None and before_snap is not None and _diff(after_snap, before_snap) > 2.5:
            return
        sleep(poll_s)


def wait_for_result_frame(dev, matcher, timeout_s: float = 8.0,
                          poll_s: float = 0.2, sleep=time.sleep,
                          now=time.monotonic, should_stop=None):
    """Return the first frame where result OK is visible, or None on timeout."""
    deadline = now() + timeout_s
    while now() < deadline and not _stop_requested(should_stop):
        frame = _nav_read(dev)
        if frame is not None and matcher.find(frame, _RESULT_BUTTON, 0.82):
            return frame
        sleep(poll_s)
    return None


def read_run_result(dev, cfg, matcher, timeout_s: float = 30.0,
                    poll_s: float = 0.5, stable_reads: int = 3,
                    settle_timeout_s: float = 14.0, sleep=time.sleep,
                    now=time.monotonic, should_stop=None) -> dict:
    """Read the post-run Result screen. Returns {coins, ingredients, read_ok}. `read_ok`
    is False when we never got a genuine Result frame (a card game / level-up modal /
    Mystery-Box screen can pre-empt or hide it — observed live: 3/15 runs returned 0 not
    because the digits misread but because the Result screen was gone by the time we read).
    A completed run cannot really yield 0 coins, so a 0 read is reported as read_ok=False —
    the caller should treat it as UNCOUNTED (banked, uncounted), never as a real 0."""
    def _out(d, ok):
        return {"coins": d["coins"], "ingredients": d["ingredients"], "read_ok": bool(ok)}

    frame = wait_for_result_frame(
        dev, matcher, timeout_s=timeout_s, poll_s=poll_s,
        sleep=sleep, now=now, should_stop=should_stop)
    if frame is None:
        return {"coins": 0, "ingredients": 0, "read_ok": False}
    best = last = read_results(frame, cfg)
    streak = 1
    polls = max(1, int(settle_timeout_s / max(poll_s, 0.01)))
    for _ in range(polls):
        if _stop_requested(should_stop):
            break
        sleep(poll_s)
        frame = _nav_read(dev)
        if frame is None or not matcher.find(frame, _RESULT_BUTTON, 0.82):
            continue
        current = read_results(frame, cfg)
        # keep the strictly-best read by (coins, then ingredients) — a later frame with
        # equal coins but a transiently-misread LOWER ingredients count must not overwrite
        # a better earlier read (which could wrongly flip read_ok to False).
        if (current["coins"], current["ingredients"]) > (best["coins"], best["ingredients"]):
            best = current
        if current == last and (current["coins"] > 0 or current["ingredients"] > 0):
            streak += 1
            if streak >= stable_reads:
                return _out(current, True)
        else:
            last = current
            streak = 1
    return _out(best, best["coins"] > 0 or best["ingredients"] > 0)


def read_wallet(dev, cfg, matcher, tries: int = 8, should_stop=None,
                sleep=time.sleep) -> "int | None":
    """Best-effort read of the menu top-bar coin balance = the GROUND-TRUTH wallet
    (survival-independent, immune to the fragile Result-screen read). Reads only when the
    menu is confirmed (Play visible) so a transient non-menu frame can't be misread, and
    only accepts a plausible positive value. Returns the balance, or None if we can't
    confidently read it. Used to reconcile a session's true net against the per-run tally."""
    region = cfg.regions.get("coin_counter") if hasattr(cfg.regions, "get") else None
    if region is None:
        return None
    for i in range(tries):
        if _stop_requested(should_stop):
            return None
        f = _nav_read(dev)
        if f is not None and matcher.find(f, "play", 0.75) is not None:
            val = read_int(f, region, cfg.templates_dir)
            if val is not None and val > 0:
                return val
        if i < tries - 1:
            _sleep_interruptible(0.2, should_stop, sleep=sleep)
    return None


def _scrolling(dev, dt: float = 0.25, thresh: float = 8.0) -> bool:
    """Big frame-to-frame diff (the whole background moving)."""
    a = _snapshot(_nav_read(dev))
    time.sleep(dt)
    b = _snapshot(_nav_read(dev))
    return a is not None and b is not None and _diff(a, b) > thresh


def _in_run(dev, matcher) -> bool:
    """A run is in progress iff the in-run HUD (the 'Slide' control button) is visible.
    This is far more reliable than frame motion: menu / loading / popup screens animate too
    (a scroll check false-positives on them, which made ensure_running return True on a
    non-run so play_until_death then ran on the menu = false ~15s deaths). Only a live run —
    including its paused 'Tap to activate Boost' start — shows the Jump/Slide controls.
    Measured: 'slide' matches 1.0 on a run, <=0.39 on menu/loading/popups."""
    f = _nav_read(dev)
    if f is None:
        return False
    return matcher.present(f, "slide", 0.72)


def _tap_template(dev, matcher, name, thresh: float = 0.72) -> bool:
    """Find a button by image and tap its centre. Returns whether it tapped."""
    f = _nav_read(dev)
    if f is None:
        return False
    pt = matcher.find(f, name, thresh)
    if pt is None:
        return False
    dev.tap(*pt)
    return True


def _visible_safe_action(frame, matcher) -> str | None:
    for name, threshold, kind in _SAFE_ACTIONS:
        if kind == "present":
            if matcher.present(frame, name, threshold):
                return name
        elif matcher.find(frame, name, threshold):
            return name
    return None


def _safe_to_back(dev, matcher) -> bool:
    """BACK on the MENU opens the Quit-game dialog; enough stray BACKs confirm it and
    kill the game (observed live: flaky adb capture returned garbage frames that matched
    NOTHING → 'unrecognized' → BACK → BACK → quit). Only allow BACK when we can POSITIVELY
    confirm a dismissable modal: a fresh, valid, non-blank frame where Play is NOT visible.
    A capture failure (None / near-uniform) or any Play match vetoes BACK — we wait instead.
    Also veto while monitor.py is solving a card game (it drops a fresh `card_active` flag):
    the card screen is NOT a dismissable popup — BACK can forfeit it and walk out of the app
    entirely (observed 2026-07-05: a template-missed 'sliding card' got BACK-spammed to the
    Android launcher). The flag is ignored after 90s so a dead monitor can't freeze nav."""
    try:
        if time.time() - os.stat(_CARD_FLAG).st_mtime < 90:
            return False                               # card game in progress -> never BACK
    except OSError:
        pass                                           # no flag -> normal popup handling
    fresh = _nav_read(dev)
    if fresh is None or float(np.std(fresh)) < 12.0:
        return False                                   # blank/stale/broken capture -> never BACK
    if matcher.find(fresh, "play", 0.70) is not None:
        return False                                   # menu Play visible -> never BACK
    return True


def _tile_checked(matcher, frame, pt) -> bool:
    """The green check sits in a tile's bottom-right corner slot; search a tile-sized ROI
    around the matched icon centre (measured: check at centre+(+52..+142,+63..+143))."""
    cx, cy = pt
    h, w = frame.shape[:2]
    roi = frame[max(0, cy - 40):min(h, cy + 200), max(0, cx - 40):min(w, cx + 240)]
    if roi.size == 0:            # pt falls outside this frame (e.g. a partial / wrong-size capture)
        return False             # -> not "checked"; the caller falls back to the sharp-adb path
    return matcher.present(roi, "tilecheck", 0.80)


def _find_stable(dev, matcher, name, thresh=0.80, tries=8, should_stop=None):
    """Poll up to `tries` fresh frames for `name`; return (pt, frame) of the first match,
    else (None, last_seen_frame). scrcpy pushes frames only on change, and the boost
    screen animates (cookie preview, coin counter, button sheen), so a SINGLE-frame
    template miss is not 'tile absent' — that false negative was aborting the whole boost
    gate and starting runs with no boosts (observed live)."""
    last = None
    for i in range(tries):
        if _stop_requested(should_stop):
            return None, last
        f = _nav_read(dev)
        if f is not None:
            last = f
            pt = matcher.find(f, name, thresh)
            if pt is not None:
                return pt, f
        if i < tries - 1:
            _sleep_interruptible(0.1, should_stop)
    return None, last


def _tile_checked_stable(dev, matcher, name, tries=6, should_stop=None) -> bool:
    """True as soon as a polled frame shows the tile AND its green check (robust to the
    check overlay lagging the tap by a frame or two)."""
    for i in range(tries):
        if _stop_requested(should_stop):
            return False
        f = _nav_read(dev)
        if f is not None:
            pt = matcher.find(f, name, 0.80)
            if pt is not None and _tile_checked(matcher, f, pt):
                return True
        if i < tries - 1:
            _sleep_interruptible(0.1, should_stop)
    return False
