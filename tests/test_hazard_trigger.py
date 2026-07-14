import threading

import numpy as np

from cookierun_bot.gestures import ACTION_JUMP, ACTION_NOOP
from cookierun_bot.policies.hazard_trigger import HazardTrigger
from cookierun_bot.policies.rule_based import ActionDecision


class _Inner:
    explore = 0.0

    def decide(self, _frame):
        return ActionDecision(ACTION_NOOP, "base/model:none:1.0")


def _async_trigger():
    h = object.__new__(HazardTrigger)
    h.inner = _Inner()
    h._async = True
    h._latest = None
    h._p_pit = 0.9
    h._p_seq = 1
    h._last_p_seq = 0
    h._state_lock = threading.Lock()
    h._frame_seq = 0
    h._generation = 0
    h._check_every = 1
    h._fcount = 0
    h._thr = 0.7
    h._below = 0
    h._above = 0
    h._ep_hits = 0
    h._confirm = 2
    h._cd_until = 0.0
    h._max_ep = 2
    h._cd_s = 0.25
    h.fires = 0
    return h


def test_async_confirmation_requires_distinct_inference_results():
    h = _async_trigger()
    frame = np.zeros((8, 8, 3), np.uint8)

    assert h.decide(frame).action == ACTION_NOOP
    assert h.decide(frame).action == ACTION_NOOP
    assert h._above == 1

    h._p_seq = 2
    assert h.decide(frame).action == ACTION_JUMP
