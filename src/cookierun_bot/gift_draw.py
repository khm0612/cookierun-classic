from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


GIFT_TEMPLATES = ("giftbtn", "giftdraw", "giftpick", "giftbox", "draw", "drawagain",
                  "giftclose", "selectmode", "cardgame")
_GIFT_BOX_FALLBACKS = (
    (0.30, 0.50),
    (0.50, 0.50),
    (0.70, 0.50),
    (0.40, 0.75),
    (0.60, 0.75),
)


@dataclass(frozen=True)
class GiftDrawResult:
    draws: int = 0
    depleted: bool = False
    opened: bool = False


def _stop_requested(should_stop=None) -> bool:
    return bool(should_stop and should_stop())


def _sleep_interruptible(seconds: float, should_stop=None,
                         sleep=time.sleep, now=time.monotonic,
                         step_s: float = 0.2) -> None:
    end = now() + seconds
    while now() < end and not _stop_requested(should_stop):
        sleep(min(step_s, end - now()))


def _snapshot(frame, max_w: int = 160, max_h: int = 90):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    sy = max(1, h // max_h)
    sx = max(1, w // max_w)
    return frame[::sy, ::sx].copy()


def _diff(a, b) -> float:
    if a.shape != b.shape:
        return 255.0
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


def _wait_for_change(dev, before, timeout_s: float = 1.0,
                     poll_s: float = 0.1, sleep=time.sleep,
                     now=time.monotonic, should_stop=None) -> None:
    before_snap = _snapshot(before)
    deadline = now() + timeout_s
    while now() < deadline and not _stop_requested(should_stop):
        after_snap = _snapshot(dev.last_frame())
        if after_snap is not None and before_snap is not None and _diff(after_snap, before_snap) > 2.5:
            return
        sleep(poll_s)


def _find(matcher, frame, name: str, threshold: float = 0.80):
    return matcher.find(frame, name, threshold)


def _present(matcher, frame, name: str, threshold: float = 0.80) -> bool:
    return matcher.present(frame, name, threshold)


def _tap_found(dev, matcher, frame, name: str, threshold: float = 0.80) -> bool:
    pt = _find(matcher, frame, name, threshold)
    if pt is None:
        return False
    dev.tap(*pt)
    return True


def _find_lower_button(matcher, frame, name: str, threshold: float = 0.80):
    h, w = frame.shape[:2]
    if h <= 120 or w <= 120:
        return _find(matcher, frame, name, threshold)
    y0, y1 = int(0.62 * h), int(0.94 * h)
    x0, x1 = int(0.42 * w), int(0.92 * w)
    pt = _find(matcher, frame[y0:y1, x0:x1], name, threshold)
    if pt is None:
        return None
    return x0 + pt[0], y0 + pt[1]


def _tap_lower_button(dev, matcher, frame, name: str, threshold: float = 0.80) -> bool:
    pt = _find_lower_button(matcher, frame, name, threshold)
    if pt is None:
        return False
    dev.tap(*pt)
    return True


def _tap_fraction(dev, frame, fx: float, fy: float) -> None:
    h, w = frame.shape[:2]
    dev.tap(int(w * fx), int(h * fy))


def _draw_again_enabled(frame) -> bool:
    h, w = frame.shape[:2]
    if h <= 120 or w <= 120:
        return True
    crop = frame[int(0.82 * h):int(0.865 * h), int(0.515 * w):int(0.555 * w)]
    if crop.size == 0:
        return True
    b, g, r = crop.reshape(-1, 3).mean(axis=0)
    return bool(b > 105 and g > 105 and b > r + 45)


def _menu_visible(matcher, frame) -> bool:
    h, w = frame.shape[:2]
    return _find(matcher, frame[int(0.62 * h):, int(0.45 * w):], "play", 0.72) is not None


def _find_gift_button(matcher, frame):
    if not _menu_visible(matcher, frame):
        return None
    pt = _find(matcher, frame, "giftbtn", 0.75)
    if pt is None:
        return None
    x, y = pt
    h, w = frame.shape[:2]
    if not (0.20 * w <= x <= 0.45 * w and y >= 0.78 * h):
        return None
    return pt


def _tap_gift_button(dev, matcher, frame) -> bool:
    pt = _find_gift_button(matcher, frame)
    if pt is None:
        return False
    x, y = pt
    # ponytail: giftbtn template is just the stable bow/top of the badge; tap lower,
    # closer to the visual center of the actual clickable present.
    dev.tap(x, min(frame.shape[0] - 1, y + int(frame.shape[0] * 0.04)))
    return True


def gift_button_visible(frame, matcher) -> bool:
    return _find_gift_button(matcher, frame) is not None


def _run_start_visible(matcher, frame) -> bool:
    if _present(matcher, frame, "slide", 0.60):
        return True
    if _present(matcher, frame, "cardgame", 0.75):
        return True
    h, w = frame.shape[:2]
    # "Select a Mode" / race chooser: reaching it means a Play path was entered. Stop.
    return _find(matcher, frame[int(0.02 * h):int(0.25 * h), int(0.28 * w):int(0.72 * w)],
                 "selectmode", 0.80) is not None


def draw_gifts(dev, matcher, log=print, should_stop=None, max_steps: int = 80,
               sleep=time.sleep, now=time.monotonic) -> GiftDrawResult:
    """Open Gift Draw and keep drawing until the game stops offering another draw.

    Entry is template-gated by ``giftbtn`` so the farm loop does not blind-tap the menu.
    Inside the gift picker, a coordinate fallback is allowed only after the picker header
    is visible; the user explicitly authorized spending Gift Draw points here.
    """
    opened = False
    saw_gift_ui = False
    idle_steps = 0
    box_index = 0
    draws = 0
    awaiting_reward_until = 0.0

    for _ in range(max_steps):
        if _stop_requested(should_stop):
            break
        frame = dev.last_frame()
        if frame is None:
            _sleep_interruptible(0.2, should_stop, sleep=sleep, now=now)
            continue

        if _run_start_visible(matcher, frame):
            log("[gift] run start detected; stopping Gift Draw")
            return GiftDrawResult(draws=draws, depleted=False, opened=opened)

        if not saw_gift_ui and _tap_found(dev, matcher, frame, "ok", 0.82):
            log("[gift] cleared result screen")
            _wait_for_change(dev, frame, timeout_s=1.5, sleep=sleep, now=now,
                             should_stop=should_stop)
            continue

        if _draw_again_enabled(frame) and _tap_lower_button(dev, matcher, frame, "drawagain", 0.80):
            log("[gift] Draw again")
            saw_gift_ui = True
            awaiting_reward_until = 0.0
            idle_steps = 0
            _sleep_interruptible(0.8, should_stop, sleep=sleep, now=now)
            continue

        if ((saw_gift_ui or not _menu_visible(matcher, frame))
                and (_tap_found(dev, matcher, frame, "confirm", 0.82)
                     or _tap_found(dev, matcher, frame, "confirm2", 0.82))):
            log("[gift] confirmed reward")
            opened = True
            saw_gift_ui = True
            idle_steps = 0
            _sleep_interruptible(0.8, should_stop, sleep=sleep, now=now)
            continue

        if (not _menu_visible(matcher, frame)
                and not _present(matcher, frame, "giftdraw", 0.75)
                and _find_lower_button(matcher, frame, "draw", 0.80) is None
                and _tap_found(dev, matcher, frame, "giftclose", 0.82)):
            log("[gift] closed reward modal")
            opened = True
            saw_gift_ui = True
            awaiting_reward_until = 0.0
            idle_steps = 0
            _wait_for_change(dev, frame, timeout_s=1.0, sleep=sleep, now=now,
                             should_stop=should_stop)
            continue

        if awaiting_reward_until > now():
            _sleep_interruptible(0.2, should_stop, sleep=sleep, now=now)
            continue
        awaiting_reward_until = 0.0

        if _present(matcher, frame, "giftpick", 0.80) or _find(matcher, frame, "giftbox", 0.75):
            saw_gift_ui = True
            if not _tap_found(dev, matcher, frame, "giftbox", 0.75):
                # ponytail: once the picker screen is verified, any box is equivalent.
                fx, fy = _GIFT_BOX_FALLBACKS[box_index % len(_GIFT_BOX_FALLBACKS)]
                _tap_fraction(dev, frame, fx, fy)
            box_index += 1
            draws += 1
            idle_steps = 0
            log(f"[gift] picked gift box ({draws})")
            awaiting_reward_until = now() + 8.0
            _wait_for_change(dev, frame, timeout_s=2.0, sleep=sleep, now=now,
                             should_stop=should_stop)
            continue

        if _tap_lower_button(dev, matcher, frame, "draw", 0.72):
            log("[gift] Draw")
            opened = True
            saw_gift_ui = True
            awaiting_reward_until = 0.0
            idle_steps = 0
            _sleep_interruptible(0.8, should_stop, sleep=sleep, now=now)
            continue

        if _tap_gift_button(dev, matcher, frame):
            log("[gift] opening Gift Draw")
            opened = True
            awaiting_reward_until = 0.0
            idle_steps = 0
            _wait_for_change(dev, frame, timeout_s=1.5, sleep=sleep, now=now,
                             should_stop=should_stop)
            continue

        if saw_gift_ui:
            idle_steps += 1
            if idle_steps >= 8:
                return GiftDrawResult(draws=draws, depleted=True, opened=opened)
            _sleep_interruptible(0.25, should_stop, sleep=sleep, now=now)
            continue

        return GiftDrawResult(draws=draws, depleted=False, opened=opened)

    return GiftDrawResult(draws=draws, depleted=False, opened=opened)
