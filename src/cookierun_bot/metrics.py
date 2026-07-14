from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RunResult:
    coins: int
    ingredients: int
    duration_s: float
    boost_cost: int = 0

    @property
    def net_coins(self) -> int:
        return self.coins - self.boost_cost


class Metrics:
    def __init__(self):
        self._runs: list[RunResult] = []
        self._unread_seconds = 0.0

    def add(self, r: RunResult) -> None:
        self._runs.append(r)

    def add_unread(self, duration_s: float) -> None:
        self._unread_seconds += duration_s

    def _total_seconds(self) -> float:
        return self._unread_seconds + sum(r.duration_s for r in self._runs)

    def runs(self) -> int:
        return len(self._runs)

    def total_coins(self) -> int:
        return sum(r.coins for r in self._runs)

    def total_boost_cost(self) -> int:
        return sum(r.boost_cost for r in self._runs)

    def total_net_coins(self) -> int:
        return sum(r.net_coins for r in self._runs)

    def last(self) -> RunResult | None:
        return self._runs[-1] if self._runs else None

    def coins_per_hour(self) -> float:
        secs = self._total_seconds()
        if secs <= 0:
            return 0.0
        return sum(r.coins for r in self._runs) / secs * 3600.0

    def ingredients_per_hour(self) -> float:
        secs = self._total_seconds()
        if secs <= 0:
            return 0.0
        return sum(r.ingredients for r in self._runs) / secs * 3600.0

    def net_coins_per_hour(self) -> float:
        secs = self._total_seconds()
        if secs <= 0:
            return 0.0
        return self.total_net_coins() / secs * 3600.0

    def summary(self) -> str:
        return (f"runs={len(self._runs)} "
                f"coins/hr={self.coins_per_hour():.0f} "
                f"net/hr={self.net_coins_per_hour():.0f} "
                f"ingredients/hr={self.ingredients_per_hour():.1f}")
