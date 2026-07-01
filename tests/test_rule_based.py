import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.policies.rule_based import RuleBasedAgent, extract_features
from cookierun_bot.gestures import ACTION_JUMP, ACTION_SLIDE, ACTION_NOOP


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {"play_area": Region(0, 0, 100, 100), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((0, 0), (0, 0), 300), RewardWeights(1, 50, 0.01, 10),
                  ["ok"], ["buy"], "templates")


def test_clear_path_is_noop():
    frame = np.full((100, 100, 3), 200, np.uint8)     # bright, no obstacles
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_NOOP


def test_ground_obstacle_triggers_jump():
    frame = np.full((100, 100, 3), 200, np.uint8)
    frame[70:100, 40:60] = 0                          # dark blob low in play area
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_JUMP


def test_overhead_only_obstacle_triggers_slide():
    frame = np.full((100, 100, 3), 200, np.uint8)
    frame[0:25, 40:60] = 0                             # dark blob only near the top
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_SLIDE
