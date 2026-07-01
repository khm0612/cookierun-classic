from cookierun_bot.metrics import RunResult, Metrics


def test_rates_computed_over_total_time():
    m = Metrics()
    m.add(RunResult(coins=100, ingredients=2, duration_s=60.0))
    m.add(RunResult(coins=200, ingredients=1, duration_s=60.0))
    # 300 coins / 120s = 2.5/s -> 9000/hr ; 3 ingredients / 120s -> 90/hr
    assert round(m.coins_per_hour(), 1) == 9000.0
    assert round(m.ingredients_per_hour(), 1) == 90.0


def test_empty_metrics_zero_rates():
    m = Metrics()
    assert m.coins_per_hour() == 0.0
    assert m.ingredients_per_hour() == 0.0
