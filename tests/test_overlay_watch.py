import numpy as np

from cookierun_bot.agents.overlay_watch import (
    SKILL_STATE,
    OverlayState,
    WarningLatch,
    channeling_score,
    state_for_action,
    state_for_frame,
)
from cookierun_bot.config import Config, Gestures, Region, RewardWeights
from cookierun_bot.gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "adb", 15, 8, "Episode 1",
                  {"play_area": Region(0, 0, 100, 100), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((1, 2), (3, 4), 300), RewardWeights(1, 50, 0.01, 10),
                  ["start"], ["buy"], "templates")


def test_overlay_shows_jump_and_slide():
    assert state_for_action(ACTION_JUMP).text == "JUMP"
    assert state_for_action(ACTION_JUMP).visible is True
    assert state_for_action(ACTION_SLIDE).text == "SLIDE"
    assert state_for_action(ACTION_SLIDE).visible is True


def test_overlay_hides_noop_by_default():
    state = state_for_action(ACTION_NOOP)
    assert state == OverlayState("", "#000000", "#ffffff", False)


def test_overlay_can_show_ready_for_noop():
    state = state_for_action(ACTION_NOOP, show_noop=True)
    assert state.text == "READY"
    assert state.visible is True


def test_warning_latch_holds_visible_action_for_reaction_delay():
    latch = WarningLatch(hold_ms=200)
    jump = state_for_action(ACTION_JUMP)
    hidden = state_for_action(ACTION_NOOP)

    assert latch.update(jump, now=1.0).text == "JUMP"
    assert latch.update(hidden, now=1.1).text == "JUMP"
    assert latch.update(hidden, now=1.3).visible is False


def test_channeling_score_detects_particle_heavy_frame():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :] = (0, 220, 255)  # BGR yellow

    assert channeling_score(frame, _cfg()) > 0.9


def test_channeling_suppresses_jump_prompt():
    class Agent:
        def act(self, frame):
            return ACTION_JUMP

    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, :] = (0, 220, 255)

    state = state_for_frame(frame, _cfg(), Agent(), channeling_threshold=0.2)

    assert state == SKILL_STATE
