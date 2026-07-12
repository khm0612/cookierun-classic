import numpy as np

from cookierun_bot.policies import hybrid_phase
from cookierun_bot.policies.hybrid_phase import HybridPhaseAgent
from cookierun_bot.policies.rule_based import ActionDecision


class StubAgent:
    def __init__(self, action, reason):
        self._action, self._reason = action, reason
        self.decides = 0
        self.observes = 0
        self.resets = 0
        self.explore = 0.0

    def decide(self, frame):
        self.decides += 1
        return ActionDecision(self._action, self._reason)

    def observe(self, frame):
        self.observes += 1

    def reset(self):
        self.resets += 1


def _frame():
    return np.zeros((270, 480, 3), np.uint8)


def test_routes_to_base_without_template(tmp_path):
    base, bonus = StubAgent(1, "model:jump:0.7"), StubAgent(2, "model:slide:0.9")
    h = HybridPhaseAgent(base, bonus, templates_dir=str(tmp_path))  # no template file
    d = h.decide(_frame())
    assert d.action == 1 and d.reason == "base/model:jump:0.7"
    assert base.decides == 1 and bonus.decides == 0
    assert bonus.observes == 1                    # idle model kept warm
    # reason prefix must keep ai_farm's reason.split(":")[1] parse working
    assert d.reason.split(":")[1] == "jump"


def test_routes_to_bonus_while_latched_and_decays(tmp_path, monkeypatch):
    base, bonus = StubAgent(1, "model:jump:0.7"), StubAgent(2, "model:slide:0.9")
    h = HybridPhaseAgent(base, bonus, templates_dir=str(tmp_path), latch_s=3.0)
    h._tpl = object()                              # pretend the template loaded
    monkeypatch.setattr(hybrid_phase, "bonustime_bgr", lambda f, t: True)
    d = h.decide(_frame())
    assert d.action == 2 and d.reason.startswith("bonus/")
    assert base.observes == 1 and bonus.decides == 1
    # banner disappears -> latch holds ~3s, then decays back to base
    monkeypatch.setattr(hybrid_phase, "bonustime_bgr", lambda f, t: False)
    h._seen -= 10.0                                # simulate latch expiry
    h._next_check = 0.0
    d2 = h.decide(_frame())
    assert d2.reason.startswith("base/")


def test_reset_and_explore_fan_out(tmp_path):
    base, bonus = StubAgent(0, "model:none:1.0"), StubAgent(0, "model:none:1.0")
    h = HybridPhaseAgent(base, bonus, templates_dir=str(tmp_path))
    h.explore = 0.25
    assert base.explore == 0.25 and bonus.explore == 0.25
    h.reset()
    assert base.resets == 1 and bonus.resets == 1
    assert h._active_name == "base"
