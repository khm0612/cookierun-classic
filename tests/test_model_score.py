"""Pure-function tests for the held-out scorer + promotion gate (scripts/model_score.py).
These exercise the event/score/gate logic with numpy only — no torch/GPU, so they run in CI.
The full score_model() (torch + demo images) is validated by running the CLI on real demos."""
import numpy as np

from model_score import extract_events, event_score, best_conf_score, gate_accepts


def test_extract_events_groups_contiguous_same_action():
    # labels: none, jump, jump, none, slide, none, jump
    labels = [0, 1, 1, 0, 2, 0, 1]
    assert extract_events(labels) == [(1, 2, 1), (4, 4, 2), (6, 6, 1)]


def test_extract_events_empty_and_all_none():
    assert extract_events([]) == []
    assert extract_events([0, 0, 0]) == []


def test_extract_events_adjacent_different_actions_split():
    # a jump immediately followed by a slide are two separate events, not one
    assert extract_events([1, 2]) == [(0, 0, 1), (1, 1, 2)]


def test_event_score_perfect_hits_no_false_fires():
    events = [(1, 2, 1), (4, 4, 2)]
    yv = np.array([0, 1, 1, 0, 2, 0])
    pred = np.array([0, 1, 1, 0, 2, 0])
    prob = np.array([0.9, 0.9, 0.9, 0.9, 0.95, 0.9])
    score, hits, fam = event_score(events, pred, prob, yv, conf=0.6)
    assert hits == 2
    assert fam == 0.0
    assert score == 1.0                      # 2/2 events hit, no false fires


def test_event_score_counts_false_fires_per_minute():
    # one none-frame fires spuriously -> fam = (1/6) * 35 * 60
    events = [(1, 1, 1)]
    yv = np.array([0, 1, 0, 0, 0, 0])
    pred = np.array([0, 1, 1, 0, 0, 0])       # index 2 is a false fire (true label none)
    prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    score, hits, fam = event_score(events, pred, prob, yv, conf=0.6, fps=35.0)
    assert hits == 1
    assert fam == (1 / 6) * 35 * 60
    assert score == 1.0 - fam / 400.0


def test_event_score_conf_gate_suppresses_low_prob_fire():
    events = [(1, 1, 1)]
    yv = np.array([0, 1, 0])
    pred = np.array([0, 1, 0])
    prob = np.array([0.5, 0.55, 0.5])         # the jump prediction is below conf 0.6
    score, hits, fam = event_score(events, pred, prob, yv, conf=0.6)
    assert hits == 0                          # gated out -> missed the event
    assert fam == 0.0


def test_best_conf_score_picks_the_maximizing_conf():
    # a low conf catches the real event but also a false fire; a high conf drops both.
    events = [(1, 1, 1)]
    yv = np.array([0, 1, 0, 0])
    pred = np.array([0, 1, 1, 0])
    prob = np.array([0.0, 0.65, 0.55, 0.0])   # real fire @0.65, false fire @0.55
    best = best_conf_score(events, pred, prob, yv, fps=35.0)
    # conf 0.6 keeps the hit and suppresses the 0.55 false fire -> best score 1.0
    assert best["conf"] == 0.6
    assert best["hits"] == 1 and best["fam"] == 0.0 and best["score"] == 1.0
    assert best["events"] == 1


def test_gate_accepts_only_on_strict_improvement():
    assert gate_accepts(0.50, 0.51) is True         # challenger beats champion
    assert gate_accepts(0.50, 0.50) is False        # tie keeps champion (no drift)
    assert gate_accepts(0.50, 0.49) is False        # regression rejected


def test_gate_margin_raises_the_bar():
    assert gate_accepts(0.50, 0.52, margin=0.01) is True    # beats by 0.02 > 0.01
    assert gate_accepts(0.50, 0.505, margin=0.01) is False  # beats by only 0.005 < margin


def test_missing_eval_demos_raises_a_catchable_exception():
    """Regression: score_model must raise a normal Exception (NOT SystemExit/BaseException) when
    no eval demos resolve, so self_farm's `except Exception` fails the gate CLOSED instead of
    killing the unattended loop."""
    import os
    import pytest
    pytest.importorskip("torch")
    from model_score import score_model

    rec = os.path.join("data", "demo")
    if not os.path.exists(os.path.join(rec, "model.pt")):
        pytest.skip("deployed model.pt not present")
    with pytest.raises(Exception):                      # RuntimeError is an Exception; SystemExit is NOT
        score_model(os.path.join(rec, "model.pt"), os.path.join(rec, "model_meta.json"),
                    eval_demos=["__no_such_demo__"])
