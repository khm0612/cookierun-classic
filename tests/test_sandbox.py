from cookierun_bot import sandbox
from cookierun_bot.gestures import ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE


def test_sandbox_policy_collects_all_coins_without_damage():
    stats = sandbox.run(("coin", "low", "coin", "overhead", "coin"))

    assert stats.coins == 3
    assert stats.damage == 0
    assert stats.actions == (
        ACTION_NOOP, ACTION_JUMP, ACTION_NOOP, ACTION_SLIDE, ACTION_NOOP,
    )


def test_low_obstacle_requires_jump():
    assert sandbox.damage_for("low", ACTION_NOOP) == 1
    assert sandbox.damage_for("low", ACTION_JUMP) == 0


def test_overhead_obstacle_requires_slide():
    assert sandbox.damage_for("overhead", ACTION_NOOP) == 1
    assert sandbox.damage_for("overhead", ACTION_SLIDE) == 0
