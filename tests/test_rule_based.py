import cv2
import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.policies.rule_based import (
    RuleBasedAgent, StreamingRuleBasedAgent, _GROUND_BAND, _OVER_BAND, _COIN_BAND,
    _FLOOR_AHEAD, _FLOOR_REF, _HAZ_REGION)
from cookierun_bot.gestures import ACTION_JUMP, ACTION_SLIDE, ACTION_NOOP


def _cfg(decision_hz=15):
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, decision_hz, "Episode 1",
                  {"play_area": Region(0, 0, 100, 100), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((0, 0), (0, 0), 300), RewardWeights(1, 50, 0.01, 10),
                  ["ok"], ["buy"], "templates")


def _frame(h=720, w=1280):
    return np.full((h, w, 3), 128, np.uint8)   # flat mid-gray: no edges, no gold


def _fill_stripes(frame, box):
    """Fill a fractional band with high-contrast vertical stripes (strong edge density)."""
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = box
    x0, y0, x1, y1 = int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)
    frame[y0:y1, x0:x1:4] = 0
    frame[y0:y1, x0 + 2:x1:4] = 255


def _fill_gold(frame, box):
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = box
    frame[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = (0, 215, 255)  # BGR gold


def _fill_green_floor(frame, box):
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = box
    y0, y1 = int(fy0 * h), int(fy1 * h)
    x0, x1 = int(fx0 * w), int(fx1 * w)
    frame[y0:y0 + max(2, (y1 - y0) // 4), x0:x1] = (40, 170, 70)


def test_clear_path_is_noop():
    agent = RuleBasedAgent(_cfg()); agent.reset()
    assert agent.act(_frame()) == ACTION_NOOP


def test_pit_ahead_triggers_jump():
    frame = _frame()                              # bright floor everywhere (ref = solid)
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = _FLOOR_AHEAD
    frame[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = (10, 40, 10)  # dark pit
    agent = RuleBasedAgent(_cfg()); agent.reset()
    assert agent.act(frame) == ACTION_JUMP


def test_dark_green_floor_gap_triggers_jump():
    frame = _dark_frame()
    _fill_green_floor(frame, _FLOOR_REF)
    agent = RuleBasedAgent(_cfg()); agent.reset()
    assert agent.act(frame) == ACTION_JUMP


def test_dark_green_floor_continues_noop_when_solid():
    frame = _dark_frame()
    _fill_green_floor(frame, _FLOOR_REF)
    _fill_green_floor(frame, _FLOOR_AHEAD)
    agent = RuleBasedAgent(_cfg()); agent.reset()
    assert agent.act(frame) == ACTION_NOOP


def test_streaming_agent_jumps_pit_immediately_once_per_frame_object():
    frame = _frame()
    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = _FLOOR_AHEAD
    frame[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = (10, 40, 10)
    agent = StreamingRuleBasedAgent(_cfg())

    decision = agent.decide(frame)

    assert decision.action == ACTION_JUMP
    assert decision.reason == "pit"
    assert agent.decide(frame).action == ACTION_NOOP


def test_streaming_agent_detects_mutated_reused_frame_object():
    frame = _frame()
    agent = StreamingRuleBasedAgent(_cfg())

    assert agent.decide(frame).reason == "clear"

    h, w = frame.shape[:2]
    fx0, fy0, fx1, fy1 = _FLOOR_AHEAD
    frame[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)] = (10, 40, 10)

    decision = agent.decide(frame)

    assert decision.action == ACTION_JUMP
    assert decision.reason == "pit"


def test_streaming_agent_requires_consecutive_hazard_frames():
    frame1 = _dark_frame()
    _paint_face(frame1, 500, 470, 640, 580)
    frame2 = frame1.copy()
    agent = StreamingRuleBasedAgent(_cfg())

    first = agent.decide(frame1)
    second = agent.decide(frame2)

    assert first.action == ACTION_NOOP
    assert first.reason == "confirming:hazard:jump"
    assert second.action == ACTION_JUMP
    assert second.reason == "hazard:jump"


def _dark_frame(h=720, w=1280):
    return np.full((h, w, 3), 40, np.uint8)           # dark forest background


def _paint_face(frame, x0, y0, x1, y1):
    """Orange blob with black jack-o-lantern 'face' pixels inside."""
    frame[y0:y1, x0:x1] = (20, 120, 255)
    fy, fx = (y0 + y1) // 2, (x0 + x1) // 2
    frame[fy - 8:fy + 8, fx - 20:fx + 20] = (10, 10, 10)


def test_ground_pumpkin_triggers_jump():
    from cookierun_bot.policies.rule_based import _orange_hazard
    h, w = 720, 1280
    frame = _dark_frame(h, w)
    # ground blob ahead: bottom at ~0.80h, inside trigger x
    _paint_face(frame, 500, 470, 640, 580)
    assert _orange_hazard(frame) == "jump"


def test_hanging_trunk_triggers_slide():
    from cookierun_bot.policies.rule_based import _orange_hazard
    h, w = 720, 1280
    frame = _dark_frame(h, w)
    # column anchored at the top of the region, tall, with a face-sized dark cutout
    frame[0:400, 480:600] = (20, 120, 255)
    frame[150:260, 495:585] = (10, 10, 10)            # jack-o-lantern face (~20% of bbox)
    assert _orange_hazard(frame) == "slide"


def test_faceless_orange_item_is_safe():
    from cookierun_bot.policies.rule_based import _orange_hazard
    h, w = 720, 1280
    frame = _dark_frame(h, w)
    frame[470:580, 500:640] = (20, 120, 255)          # smooth orange blob, NO black face
    assert _orange_hazard(frame) is None


def test_orange_background_zone_does_not_fire():
    from cookierun_bot.policies.rule_based import _orange_hazard
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :] = (30, 140, 230)                      # bright orange everywhere (oven/desert)
    assert _orange_hazard(frame) is None


def test_edge_clutter_does_not_cause_actions():
    # Generic edge density fires on clutter, not hazards; measured live it caused jump spam
    # (360 jumps/201s) — the disciplined policy must NOT act on it.
    for band in (_GROUND_BAND, _OVER_BAND):
        frame = _frame(); _fill_stripes(frame, band)
        agent = RuleBasedAgent(_cfg()); agent.reset()
        assert agent.act(frame) == ACTION_NOOP


def test_coins_ignored_by_default():
    frame = _frame(); _fill_gold(frame, _COIN_BAND)
    agent = RuleBasedAgent(_cfg()); agent.reset()
    assert agent.act(frame) == ACTION_NOOP      # seek_coins defaults OFF (magnet collects)


def test_coin_jump_optin_has_cooldown():
    frame = _frame(); _fill_gold(frame, _COIN_BAND)
    agent = RuleBasedAgent(_cfg(), seek_coins=True); agent.reset()
    assert agent.act(frame) == ACTION_JUMP      # first: jump for coins
    assert agent.act(frame) == ACTION_NOOP      # cooldown suppresses immediate re-jump


def test_jump_cooldown_scales_with_decision_hz():
    frame = _frame(); _fill_gold(frame, _COIN_BAND)
    agent = RuleBasedAgent(_cfg(decision_hz=20), seek_coins=True); agent.reset()
    assert agent.act(frame) == ACTION_JUMP
    for _ in range(15):
        assert agent.act(frame) == ACTION_NOOP


def test_real_frame_derived_policy_cases():
    ground = _dark_frame()
    _paint_face(ground, 500, 470, 640, 580)
    hanging = _dark_frame()
    hanging[0:400, 480:600] = (20, 120, 255)
    hanging[150:260, 495:585] = (10, 10, 10)
    safe_item = _dark_frame()
    safe_item[470:580, 500:640] = (20, 120, 255)

    cases = [
        ("dark forest pumpkin", ground, ACTION_JUMP),
        ("hanging trunk", hanging, ACTION_SLIDE),
        ("faceless orange item", safe_item, ACTION_NOOP),
    ]
    for name, frame, expected in cases:
        agent = RuleBasedAgent(_cfg())
        agent.reset()
        assert agent.act(frame) == expected, name


# --- taxonomy hazard suite (synthetic frames built from the vision-workflow recipes) ---

def _bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(
        np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def test_ice_tower_triggers_jump():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[330:600, 500:610] = _bgr(95, 90, 215)       # pale-cyan tower on the ground
    frame[420:500, 530:580] = (10, 10, 10)            # black cat body inside
    assert _hazard(frame) == "jump"


def test_ice_tower_without_cat_is_ignored():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[330:600, 500:610] = _bgr(95, 90, 215)       # cyan blob but NO cat => bg candy hill
    assert _hazard(frame) is None


def _paint_hedgehog(frame, body_hsv):
    body = _bgr(*body_hsv)
    frame[520:580, 500:640] = body                    # body (140x105 with spikes = in-gates)
    for sx in (500, 540, 580, 620):                   # spike triangles on top
        cv2.fillPoly(frame, [np.array([[sx, 520], [sx + 20, 520], [sx + 10, 475]])], body)
    frame[535:565, 550:585] = _bgr(20, 80, 210)       # pale face patch (S<110, V>180)


def test_hedgehog_triggers_jump_both_colourways():
    from cookierun_bot.policies.rule_based import _hazard
    for hsv_body in ((174, 160, 120), (12, 200, 130)):  # wraparound maroon + auburn brown
        frame = _dark_frame()
        _paint_hedgehog(frame, hsv_body)
        assert _hazard(frame) == "jump", hsv_body


def test_smooth_maroon_blob_is_ignored():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[490:595, 500:640] = _bgr(174, 160, 120)     # rounded bun: no spikes, no face
    assert _hazard(frame) is None


def _paint_scissor_blade(frame):
    # top-anchored cream blade, wide at the ceiling and tapering downward (bot/top <= 0.55)
    cv2.fillPoly(frame, [np.array([[500, 0], [660, 0], [600, 300], [560, 300]])],
                 _bgr(22, 65, 230))


def test_scissor_blades_trigger_slide():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    _paint_scissor_blade(frame)
    assert _hazard(frame) == "slide"


def test_rectangular_cream_column_is_ignored():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[0:250, 500:630] = _bgr(22, 65, 230)         # no downward taper => banner/light shaft
    assert _hazard(frame) is None


def test_rock_spikes_trigger_jump():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    cv2.fillPoly(frame, [np.array([[500, 600], [640, 600], [570, 470]])], (180, 180, 180))
    frame[600:615, 500:640] = _bgr(50, 150, 150)      # green grass tufts at the base
    assert _hazard(frame) == "jump"


def test_grey_shape_without_tufts_is_ignored():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    cv2.fillPoly(frame, [np.array([[500, 600], [640, 600], [570, 470]])], (180, 180, 180))
    assert _hazard(frame) is None                     # background mountain: no grass base


def test_falling_pins_trigger_jump():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[300:360, 520:540] = (240, 240, 240)         # two thin white pins falling together
    frame[300:360, 600:620] = (240, 240, 240)
    assert _hazard(frame) == "jump"


def test_single_pin_is_ignored():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    frame[300:360, 520:540] = (240, 240, 240)         # one bar could be a UI glyph
    assert _hazard(frame) is None


def test_round_silver_coin_is_not_a_pin():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    cv2.circle(frame, (550, 330), 30, (240, 240, 240), -1)   # round coin, same palette
    cv2.circle(frame, (650, 330), 30, (240, 240, 240), -1)
    assert _hazard(frame) is None


def test_scissors_win_priority_over_jump_hazards():
    from cookierun_bot.policies.rule_based import _hazard
    frame = _dark_frame()
    _paint_scissor_blade(frame)                       # scissors overhead (slide, never jump)
    frame[300:360, 700:720] = (240, 240, 240)         # plus pin-like bars nearer the cookie
    frame[300:360, 760:780] = (240, 240, 240)
    assert _hazard(frame) == "slide"


# --- BONUSTIME suppression gate (letter-tracker prefix rule) ---

def _paint_banner(frame, states):
    """Paint the 9 letter cells of the BONUSTIME tracker: 'L' = lit (saturated red),
    'G' = grey."""
    from cookierun_bot.policies.rule_based import _BONUS_BOX
    h, w = frame.shape[:2]
    x0, y0 = int(_BONUS_BOX[0] * w), int(_BONUS_BOX[1] * h)
    x1, y1 = int(_BONUS_BOX[2] * w), int(_BONUS_BOX[3] * h)
    frame[y0:y1, x0:x1] = (100, 100, 100)             # grey banner base (S=0 => not lit)
    cw = (x1 - x0) / 9.0
    for i, s in enumerate(states):
        if s == "L":
            frame[y0:y1, int(x0 + i * cw):int(x0 + (i + 1) * cw)] = (0, 0, 255)


def test_bonus_drain_prefix_suppresses_hazards():
    from cookierun_bot.policies.rule_based import _bonus_active, _hazard
    frame = _dark_frame()
    _paint_face(frame, 500, 470, 640, 580)            # a pumpkin that would fire 'jump'
    _paint_banner(frame, "LLLLLGGGG")                 # bonus drain: contiguous 5-prefix
    assert _bonus_active(frame) is True
    assert _hazard(frame) is None                     # suppressed during the bonus stage


def test_random_collection_pattern_does_not_suppress():
    from cookierun_bot.policies.rule_based import _bonus_active, _hazard
    frame = _dark_frame()
    _paint_face(frame, 500, 470, 640, 580)
    _paint_banner(frame, "LGLLGLGGG")                 # live play: random letters collected
    assert _bonus_active(frame) is False
    assert _hazard(frame) == "jump"                   # hazards stay armed in live play


def test_all_grey_banner_does_not_suppress():
    from cookierun_bot.policies.rule_based import _bonus_active
    frame = _dark_frame()
    _paint_banner(frame, "GGGGGGGGG")                 # zero letters collected = live play
    assert _bonus_active(frame) is False
