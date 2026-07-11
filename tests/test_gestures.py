import random

from cookierun_bot.config import Gestures
from cookierun_bot.gestures import (SlideHold, apply_action, ACTION_NOOP, ACTION_JUMP,
                                    ACTION_SLIDE)


_JITTER = dict(tap_jitter_px=55, hold_jitter_frac=0.15)


def test_noop_does_nothing(fake_device):
    apply_action(fake_device, ACTION_NOOP, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.taps == [] and fake_device.holds == []


def test_jump_holds_jump_button_for_higher_arc(fake_device):
    apply_action(fake_device, ACTION_JUMP, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.holds == [(200, 1600, 250)]   # default jump_hold_ms


def test_jump_taps_when_hold_disabled(fake_device):
    g = Gestures((200, 1600), (880, 1600), 300, jump_hold_ms=0)
    apply_action(fake_device, ACTION_JUMP, g)
    assert fake_device.taps == [(200, 1600)]


def test_slide_holds_slide_button(fake_device):
    apply_action(fake_device, ACTION_SLIDE, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.holds == [(880, 1600, 300)]


def test_jump_jitter_stays_inside_button_and_hold_bounds(fake_device):
    g = Gestures((200, 1600), (880, 1600), 300, **_JITTER)
    random.seed(0)
    for _ in range(300):
        apply_action(fake_device, ACTION_JUMP, g)
    for x, y, ms in fake_device.holds:
        assert 200 - 55 <= x <= 200 + 55        # clamped to the jitter radius
        assert 1600 - 55 <= y <= 1600 + 55
        assert 211 <= ms <= 289                 # jump_hold 250 +/- 15%


def test_slide_jitter_stays_inside_button_and_hold_bounds(fake_device):
    g = Gestures((200, 1600), (880, 1600), 300, **_JITTER)
    random.seed(1)
    for _ in range(300):
        apply_action(fake_device, ACTION_SLIDE, g)
    for x, y, ms in fake_device.holds:
        assert 880 - 55 <= x <= 880 + 55
        assert 1600 - 55 <= y <= 1600 + 55
        assert 254 <= ms <= 346                 # slide_hold 300 +/- 15%


def test_jitter_actually_varies_never_repeats_a_pixel(fake_device):
    g = Gestures((200, 1600), (880, 1600), 300, **_JITTER)
    random.seed(2)
    for _ in range(50):
        apply_action(fake_device, ACTION_JUMP, g)
    xs = {x for x, _, _ in fake_device.holds}
    ms = {m for _, _, m in fake_device.holds}
    assert len(xs) > 5 and len(ms) > 3          # not one repeated pixel/duration = the whole point


def test_jitter_off_by_default_is_deterministic(fake_device):
    for _ in range(5):
        apply_action(fake_device, ACTION_JUMP, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.holds == [(200, 1600, 250)] * 5   # default (off) never scatters


# ---- SlideHold: variable-length slide via press/release ----------------------------

class _PressDevice:
    """Fake with the DOWN/UP primitive (LDPlayer path)."""
    def __init__(self):
        self.presses, self.releases, self.holds = [], [], []
    def press(self, x, y): self.presses.append((x, y))
    def release(self, x, y): self.releases.append((x, y))
    def hold(self, x, y, ms): self.holds.append((x, y, ms))


_G = Gestures((200, 1600), (880, 1600), 300)


def test_slidehold_presses_once_per_span_and_releases_after_grace():
    d, s = _PressDevice(), SlideHold(grace_s=0.2)
    s.update(d, _G, True, now=0.0)                    # DOWN
    for t in (0.02, 0.05, 0.5, 1.0):                  # sustained prediction: NO re-press
        s.update(d, _G, True, now=t)
    assert d.presses == [(880, 1600)] and d.releases == [] and s.held
    s.update(d, _G, False, now=1.1)                   # within grace: still held
    assert s.held
    s.update(d, _G, False, now=1.25)                  # grace expired -> UP
    assert not s.held and len(d.releases) == 1 and d.holds == []


def test_slidehold_no_time_cap_holds_through_a_long_continuous_slide():
    d, s = _PressDevice(), SlideHold(grace_s=0.2)
    s.update(d, _G, True, now=0.0)                    # DOWN
    for t in (2.0, 5.0, 8.0, 12.0):                  # 12s continuous slide: the game allows it
        s.update(d, _G, True, now=t)
    assert d.presses == [(880, 1600)] and d.releases == [] and s.held   # ONE press, never blipped
    s.update(d, _G, False, now=12.25)                # stopped predicting -> lift after grace only
    assert not s.held and len(d.releases) == 1


def test_slidehold_min_hold_keeps_slide_down_through_a_brief_prediction():
    """min_hold_s stops the 'slide too short' stutter: a slide the model predicts for only an
    instant is still HELD for at least min_hold_s, so the cookie doesn't pop up mid-obstacle."""
    d, s = _PressDevice(), SlideHold(grace_s=0.2, min_hold_s=0.45)
    s.update(d, _G, True, now=0.0)                     # DOWN (model predicts slide for one frame)
    s.update(d, _G, False, now=0.05)                  # model already stopped predicting slide
    s.update(d, _G, False, now=0.30)                  # PAST grace (0.25) but BEFORE min_hold -> held
    assert s.held and d.releases == []
    s.update(d, _G, False, now=0.50)                  # past min_hold AND grace -> UP
    assert not s.held and len(d.releases) == 1


def test_slidehold_release_interrupts_a_min_hold_slide_so_a_jump_never_blocks():
    """A JUMP calls release() to free the one finger; release() must lift IMMEDIATELY even mid
    min-hold, so a 1.5s min-hold slide can never block a needed jump."""
    d, s = _PressDevice(), SlideHold(grace_s=0.4, min_hold_s=1.5)
    s.update(d, _G, True, now=0.0)                     # DOWN, min-hold would run to 1.5s
    s.release(d, _G)                                   # jump path frees the finger at t~0
    assert not s.held and len(d.releases) == 1         # lifted immediately, not held to 1.5s


def test_slidehold_min_hold_does_not_shorten_a_genuinely_long_slide():
    """A slide the model wants for longer than min_hold is NOT clipped at min_hold — it holds
    while predicted, plus grace (min_hold is a floor, never a cap)."""
    d, s = _PressDevice(), SlideHold(grace_s=0.2, min_hold_s=0.45)
    s.update(d, _G, True, now=0.0)
    for t in (0.3, 0.6, 1.0):                          # predicted well past the 0.45 min_hold
        s.update(d, _G, True, now=t)
    assert s.held and d.releases == []
    s.update(d, _G, False, now=1.25)                  # grace after the last predict -> UP
    assert not s.held and len(d.releases) == 1


def test_slidehold_release_is_idempotent_and_lifts_near_the_down_point():
    d, s = _PressDevice(), SlideHold()
    s.update(d, _G, True, now=0.0)
    s.release(d, _G)
    s.release(d, _G)                                  # double release: one UP only
    assert len(d.releases) == 1
    (px, py), (rx, ry) = d.presses[0], d.releases[0]
    assert abs(rx - px) <= 4 and abs(ry - py) <= 4    # finger lifts where it landed


def test_force_release_lifts_finger_even_after_a_lost_up_cleared_held():
    d, s = _PressDevice(), SlideHold()
    s.update(d, _G, True, now=0.0)                    # DOWN
    s.held = False                                    # simulate a lost/rejected UP: state
    #                                                   says not-held, but finger is stuck
    s.force_release(d, _G)                            # boundary cleanup MUST still lift it
    assert len(d.releases) == 1 and not s.held


def test_force_release_is_a_safe_noop_when_nothing_was_pressed():
    d, s = _PressDevice(), SlideHold()
    s.force_release(d, _G)                            # never pressed -> still sends one UP
    assert len(d.releases) == 1 and d.presses == []   # (stray UP is harmless on Android)


def test_slidehold_falls_back_to_single_fixed_hold_without_press(fake_device):
    s = SlideHold(grace_s=0.2)
    for t in (0.0, 0.05, 0.1):                        # span start + refires
        s.update(fake_device, _G, True, now=t)
    assert fake_device.holds == [(880, 1600, 300)]    # ONE fixed hold, not one per tick
    s.update(fake_device, _G, False, now=0.5)         # grace expiry just clears the flag
    assert not s.held and len(fake_device.holds) == 1
