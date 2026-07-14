# ponytail: used only by env.py's CookieRunEnv (the planned-RL scaffold), not by the shipped
# behavioral-cloning pipeline. Intentional — see the note atop env.py before deleting.
from __future__ import annotations


class RewardTracker:
    def __init__(self, w):
        self._w = w
        self.reset()

    def reset(self) -> None:
        self._prev_coins = 0
        self._prev_boxes = 0
        self._total_coins = 0
        self._total_boxes = 0
        self._steps = 0

    def update(self, coins, boxes: int, dead: bool) -> float:
        self._steps += 1
        coin_delta = 0
        if coins is not None and coins >= self._prev_coins:
            coin_delta = coins - self._prev_coins
            self._prev_coins = coins
            self._total_coins = coins
        box_delta = 0
        if boxes >= self._prev_boxes:
            box_delta = boxes - self._prev_boxes
            self._prev_boxes = boxes
            self._total_boxes = boxes

        reward = self._w.w_coin * coin_delta + self._w.w_box * box_delta
        if dead:
            reward -= self._w.death_penalty
        else:
            reward += self._w.w_survive
        return reward

    def summary(self) -> dict:
        return {"coins": self._total_coins, "boxes": self._total_boxes,
                "steps": self._steps}
