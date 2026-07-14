from cookierun_bot.config import RewardWeights
from cookierun_bot.reward import RewardTracker


def _w():
    return RewardWeights(w_coin=1.0, w_box=50.0, w_survive=0.01, death_penalty=10.0)


def test_coin_delta_rewarded_not_absolute():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=10, boxes=0, dead=False) == 10 * 1.0 + 0.01   # first delta from 0
    assert rt.update(coins=15, boxes=0, dead=False) == 5 * 1.0 + 0.01    # +5 more
    assert rt.update(coins=15, boxes=0, dead=False) == 0.0 + 0.01        # no new coins


def test_box_pickup_gives_big_bonus():
    rt = RewardTracker(_w())
    rt.reset()
    r = rt.update(coins=0, boxes=1, dead=False)
    assert r == 50.0 + 0.01


def test_none_coins_counts_zero_delta():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=None, boxes=0, dead=False) == 0.01


def test_ocr_regression_does_not_double_award_on_recovery():
    rt = RewardTracker(_w())
    rt.reset()

    rt.update(coins=100, boxes=5, dead=False)
    assert rt.update(coins=40, boxes=2, dead=False) == 0.01
    assert rt.update(coins=105, boxes=6, dead=False) == 5.0 + 50.0 + 0.01


def test_death_applies_penalty_and_no_survive_bonus():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=0, boxes=0, dead=True) == -10.0


def test_summary_tracks_totals():
    rt = RewardTracker(_w())
    rt.reset()
    rt.update(coins=20, boxes=1, dead=False)
    s = rt.summary()
    assert s["coins"] == 20 and s["boxes"] == 1 and s["steps"] == 1
