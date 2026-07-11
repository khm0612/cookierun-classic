from __future__ import annotations
import random
import time

ACTION_NOOP = 0
ACTION_JUMP = 1
ACTION_SLIDE = 2
N_ACTIONS = 3


def _jitter_point(button, g) -> tuple[int, int]:
    """Humanize the tap position. Tapping the SAME exact pixel hundreds of times per run is
    a trivial server-side bot tell, so scatter the point with a Gaussian clustered near the
    button centre (like a human thumb), clamped to `tap_jitter_px` so it always lands inside
    the large on-screen Jump/Slide zone. jitter_px<=0 -> exact centre (deterministic)."""
    r = getattr(g, "tap_jitter_px", 0)
    if not r or r <= 0:
        return int(button[0]), int(button[1])
    dx = max(-r, min(r, random.gauss(0, r * 0.5)))
    dy = max(-r, min(r, random.gauss(0, r * 0.5)))
    return int(button[0] + dx), int(button[1] + dy)


def _jitter_hold(hold_ms: int, g) -> int:
    """Humanize the hold DURATION too — a fixed millisecond hold is as robotic as a fixed
    pixel. +/- `hold_jitter_frac`, bounded so slide/jump mechanics stay intact. frac<=0 ->
    exact (deterministic)."""
    frac = getattr(g, "hold_jitter_frac", 0.0)
    if not frac or frac <= 0:
        return int(hold_ms)
    return max(1, int(round(hold_ms * (1.0 + random.uniform(-frac, frac)))))


class SlideHold:
    """Variable-length slide: press DOWN when the model starts predicting slide, hold while
    it keeps predicting (grace_s bridges single-frame flickers), release UP grace_s after it
    stops. CookieRun lets you hold slide indefinitely (user-confirmed), so there is
    deliberately NO time cap — the finger stays down exactly as long as the model wants slide,
    and a genuine long slide tunnel is never clipped (a hard cap would blip the finger up
    mid-tunnel and stand the cookie into an obstacle). A pathological 'stuck on slide' can't
    hang a run either: a stuck prediction means a frozen/OOD capture, which the play loop
    already ends via its HUD-absent / stall-tap / run-boundary releases (each forces the
    finger up). The old fixed slide_hold_ms swipe stood the cookie up mid-obstacle, and every
    70fps re-fire queued another full-length swipe in the adb shell (seconds of input
    backlog). Devices without press/release (scrcpy/network/fakes) fall back to ONE
    fixed-length hold per span instead of a hold per decision tick."""

    def __init__(self, grace_s: float = 0.20, min_hold_s: float = 0.0):
        self._grace = grace_s
        self._min_hold = min_hold_s     # once a slide starts, hold >= this long (anti stutter-slide)
        self.held = False
        self._deadline = 0.0
        self._start = 0.0
        self._pt = (0, 0)

    def update(self, device, g, want_slide: bool, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        if want_slide:
            if not self.held:
                self.held = True
                self._start = now
                self._pt = _jitter_point(g.slide_button, g)
                if hasattr(device, "press"):
                    device.press(*self._pt)
                else:               # no DOWN/UP primitive: one fixed hold per span
                    device.hold(self._pt[0], self._pt[1],
                                _jitter_hold(g.slide_hold_ms, g))
            self._deadline = now + self._grace
        # lift only once the model has stopped wanting slide for grace_s AND the minimum hold has
        # elapsed — the min-hold keeps the cookie DOWN through the whole obstacle instead of
        # popping up after a few frames (the "slide too short" stutter the user observed live).
        if self.held and now >= self._deadline and now >= self._start + self._min_hold:
            self.release(device, g)

    def release(self, device, g) -> None:
        if not self.held:
            return
        self.held = False
        if hasattr(device, "press"):
            # tiny lift-off drift: a real finger never lifts on the exact landing pixel
            device.release(self._pt[0] + random.randint(-4, 4),
                           self._pt[1] + random.randint(-4, 4))

    def force_release(self, device, g) -> None:
        """Unconditionally lift the finger at a run boundary. If a normal release()'s UP was
        silently rejected by adb (held cleared but the pointer physically stuck DOWN), the
        finger would perma-slide the NEXT run and jam every menu tap in between; this re-sends
        UP regardless of `held`. A stray UP with no matching DOWN is a no-op, so it is always
        safe to call — even at loop start to clear an orphan a crashed prior run left behind."""
        self.held = False
        if hasattr(device, "press"):
            x, y = self._pt if self._pt != (0, 0) else (int(g.slide_button[0]),
                                                        int(g.slide_button[1]))
            device.release(x, y)


def apply_action(device, action: int, g) -> None:
    if action == ACTION_JUMP:
        x, y = _jitter_point(g.jump_button, g)
        # holding Jump jumps higher/longer than a tap — use a held press when configured
        hold_ms = getattr(g, "jump_hold_ms", 0)
        if hold_ms > 0:
            device.hold(x, y, _jitter_hold(hold_ms, g))
        else:
            device.tap(x, y)
    elif action == ACTION_SLIDE:
        x, y = _jitter_point(g.slide_button, g)
        device.hold(x, y, _jitter_hold(g.slide_hold_ms, g))
    # ACTION_NOOP: intentionally do nothing
