"""HybridAgent = LearnedAgent (primary) + CV hazard overrides only when the model is passive."""
import cookierun_bot.policies.hybrid as hy
from cookierun_bot.policies.hybrid import HybridAgent
from cookierun_bot.policies.rule_based import ActionDecision
from cookierun_bot.gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE

NONE = ActionDecision(ACTION_NOOP, "model:none:1.00")
JUMP = ActionDecision(ACTION_JUMP, "model:jump:0.80")
FRAME = object()   # detectors are monkeypatched, so the frame value is irrelevant


class FakeLearned:
    def __init__(self, decisions):
        self.decisions, self.i, self.reset_called, self._device = decisions, 0, 0, "cpu"
    def decide(self, frame):
        d = self.decisions[min(self.i, len(self.decisions) - 1)]; self.i += 1; return d
    def reset(self):
        self.reset_called += 1


def _mk(monkeypatch, decisions, pit=False, hazard=None, **kw):
    monkeypatch.setattr(hy, "_pit_ahead", lambda f: pit)
    monkeypatch.setattr(hy, "_hazard", lambda f: hazard)
    return HybridAgent(cfg=None, learned=FakeLearned(decisions), **kw)


def test_trusts_the_model_when_it_acts(monkeypatch):
    a = _mk(monkeypatch, [JUMP], pit=True, hazard="slide")   # detectors would fire...
    d = a.decide(FRAME, now=1.0)
    assert d.action == ACTION_JUMP and d.reason == "model:jump:0.80"   # ...but model acted -> trust it


def test_pit_override_when_model_passive(monkeypatch):
    a = _mk(monkeypatch, [NONE], pit=True)
    d = a.decide(FRAME, now=1.0)
    assert d.action == ACTION_JUMP and d.reason == "cv:pit"


def test_slide_override_for_overhead_hazard(monkeypatch):
    a = _mk(monkeypatch, [NONE], pit=False, hazard="slide")
    d = a.decide(FRAME, now=1.0)
    assert d.action == ACTION_SLIDE and d.reason == "cv:slide"


def test_no_override_when_nothing_detected(monkeypatch):
    a = _mk(monkeypatch, [NONE], pit=False, hazard=None)
    assert a.decide(FRAME, now=1.0).action == ACTION_NOOP


def test_pit_beats_hazard_priority(monkeypatch):
    a = _mk(monkeypatch, [NONE], pit=True, hazard="slide")
    assert a.decide(FRAME, now=1.0).reason == "cv:pit"


def test_jump_override_off_by_default(monkeypatch):
    a = _mk(monkeypatch, [NONE, NONE], pit=False, hazard="jump")   # default overrides = pit+slide
    assert a.decide(FRAME, now=1.0).action == ACTION_NOOP
    b = _mk(monkeypatch, [NONE], pit=False, hazard="jump", overrides=("pit", "slide", "jump"))
    assert b.decide(FRAME, now=1.0).reason == "cv:jump"


def test_jump_override_cooldown_blocks_spam(monkeypatch):
    a = _mk(monkeypatch, [NONE, NONE, NONE], pit=True, override_cd_s=0.35, cv_hz=1000)
    assert a.decide(FRAME, now=1.00).reason == "cv:pit"       # fires
    assert a.decide(FRAME, now=1.10).action == ACTION_NOOP    # within cooldown -> suppressed
    assert a.decide(FRAME, now=1.40).reason == "cv:pit"       # cooldown elapsed -> fires again


def test_slide_override_is_sustained_not_cooldowned(monkeypatch):
    # a persistent overhead hazard must re-assert every tick so SlideHold keeps the finger down
    a = _mk(monkeypatch, [NONE, NONE, NONE], hazard="slide", cv_hz=1000)
    assert a.decide(FRAME, now=1.00).reason == "cv:slide"
    assert a.decide(FRAME, now=1.01).reason == "cv:slide"     # no cooldown gap
    assert a.decide(FRAME, now=1.02).reason == "cv:slide"


def test_cv_throttle_skips_work_between_ticks(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(hy, "_pit_ahead", lambda f: (calls.__setitem__("n", calls["n"] + 1), True)[1])
    monkeypatch.setattr(hy, "_hazard", lambda f: None)
    a = HybridAgent(cfg=None, learned=FakeLearned([NONE, NONE, NONE]), cv_hz=30.0)
    a.decide(FRAME, now=1.000)          # runs CV (fires pit)
    a.decide(FRAME, now=1.005)          # <33ms later -> throttled, no CV call
    assert calls["n"] == 1
    a.decide(FRAME, now=1.100)          # >33ms -> CV runs again (but pit-cooldown may gate)


def test_reset_clears_cooldown_throttle_and_forwards(monkeypatch):
    a = _mk(monkeypatch, [NONE, NONE], pit=True, cv_hz=1000)
    a.decide(FRAME, now=1.0)            # sets cooldown
    a.reset()
    assert a._learned.reset_called == 1 and a._cd_until == 0.0 and a._last_cv == 0.0
    assert a.decide(FRAME, now=1.0).reason == "cv:pit"    # fires immediately post-reset
